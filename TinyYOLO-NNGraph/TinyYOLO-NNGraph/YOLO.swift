import MetalPerformanceShaders
import QuartzCore

class YOLO {
  public static let inputWidth = 416
  public static let inputHeight = 416

  struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }

  struct Result {
    var predictions = [Prediction]()
    var debugTexture: MTLTexture?
    var elapsed: CFTimeInterval = 0
  }

  let commandQueue: MTLCommandQueue
  let graph: MPSNNGraph
  let lanczos: MPSImageLanczosScale
  let scaledImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 416, height: 416, featureChannels: 3)

  public init(commandQueue: MTLCommandQueue) {
    self.commandQueue = commandQueue

    // Scales the image to 416x416 pixels.
    let device = commandQueue.device
    lanczos = MPSImageLanczosScale(device: device)

    // Create a placeholder for the input image.
    // Note: YOLO expects the input pixels to be in the range 0-1. Our input
    // texture most likely has pixels with values 0-255. However, since its
    // pixel format is .unorm8 and the channel format for the graph's input
    // image is .float16, Metal will automatically convert the pixels to be
    // between 0 and 1.
    let inputImage = MPSNNImageNode(handle: nil)

    let conv1 = MPSCNNConvolutionNode(source: inputImage,
                                      weights: DataSource("conv1", 3, 3, 3, 16))

    let pool1 = MPSCNNPoolingMaxNode(source: conv1.resultImage, filterSize: 2)

    let conv2 = MPSCNNConvolutionNode(source: pool1.resultImage,
                                      weights: DataSource("conv2", 3, 3, 16, 32))

    let pool2 = MPSCNNPoolingMaxNode(source: conv2.resultImage, filterSize: 2)

    let conv3 = MPSCNNConvolutionNode(source: pool2.resultImage,
                                      weights: DataSource("conv3", 3, 3, 32, 64))

    let pool3 = MPSCNNPoolingMaxNode(source: conv3.resultImage, filterSize: 2)

    let conv4 = MPSCNNConvolutionNode(source: pool3.resultImage,
                                      weights: DataSource("conv4", 3, 3, 64, 128))

    let pool4 = MPSCNNPoolingMaxNode(source: conv4.resultImage, filterSize: 2)

    let conv5 = MPSCNNConvolutionNode(source: pool4.resultImage,
                                      weights: DataSource("conv5", 3, 3, 128, 256))

    let pool5 = MPSCNNPoolingMaxNode(source: conv5.resultImage, filterSize: 2)

    let conv6 = MPSCNNConvolutionNode(source: pool5.resultImage,
                                      weights: DataSource("conv6", 3, 3, 256, 512))

    // The pool6 layer is slightly different from the other pooling layers.
    // It has stride 1, so it doesn't actually make the image any smaller.
    // Because we want it to use "clamp" mode, it gets its own padding policy.
    let pool6 = MPSCNNPoolingMaxNode(source: conv6.resultImage, filterSize: 2, stride: 1)
    pool6.paddingPolicy = Pool6Padding()

    let conv7 = MPSCNNConvolutionNode(source: pool6.resultImage,
                                      weights: DataSource("conv7", 3, 3, 512, 1024))

    let conv8 = MPSCNNConvolutionNode(source: conv7.resultImage,
                                      weights: DataSource("conv8", 3, 3, 1024, 1024))

    // The final convolution layer uses a 1x1 kernel and no activation function.
    let conv9 = MPSCNNConvolutionNode(source: conv8.resultImage,
                                      weights: DataSource("conv9", 1, 1, 1024, 125, useLeaky: false))

    if let graph = MPSNNGraph(device: device, resultImage: conv9.resultImage) {
      self.graph = graph
    } else {
      fatalError("Error: could not initialize graph")
    }

    // Enable extra debugging output.
    graph.options = .verbose
    print(graph.debugDescription)
  }

  public func predict(texture: MTLTexture, completionHandler handler: @escaping (Result) -> Void) {
    let startTime = CACurrentMediaTime()
    let commandBuffer = commandQueue.makeCommandBuffer()

    // For debugging purposes, we can show the scaled image on the main UI.
    // But to do that it must be a real MPSImage, not a temporary one.
    //let scaledImg = MPSImage(device: commandQueue.device, imageDescriptor: scaledImgDesc)

    let scaledImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: scaledImgDesc)
    lanczos.encode(commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: scaledImg.texture)

    let outputImg = graph.encode(to: commandBuffer, sourceImages: [scaledImg])

    commandBuffer.addCompletedHandler { [outputImg] commandBuffer in
      var result = Result()
      if commandBuffer.status == .completed {
        result.predictions = self.computeBoundingBoxes(outputImg)
      }
      result.elapsed = CACurrentMediaTime() - startTime
      //result.debugTexture = scaledImg.texture
      handler(result)
    }

    commandBuffer.commit()

    /*
    // Can't use the executeAsync API because we need to resize the texture
    // first and the graph does not have a resize node.
    graph.executeAsync(withSourceImages: [input]) { image, status, error in
      print(status, error, Thread.isMainThread)

      if let image = image {
        handler(self.computeBoundingBoxes(image))
      } else {
        handler([])
      }
    }
    */
  }

  // This function is exactly the same as in Forge's YOLO example.
  func computeBoundingBoxes(_ image: MPSImage) -> [Prediction] {
    let features = image.toFloatArray()
    assert(features.count == 13*13*128)

    // We only run the convolutional part of YOLO on the GPU. The last part of
    // the process is done on the CPU. It should be possible to do this on the
    // GPU too, but it might not be worth the effort.

    var predictions = [Prediction]()

    let blockSize: Float = 32
    let gridHeight = 13
    let gridWidth = 13
    let boxesPerCell = 5
    let numClasses = 20

    // This helper function finds the offset in the features array for a given
    // channel for a particular pixel. (See the comment below.)
    func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
      let slice = channel / 4
      let indexInSlice = channel - slice*4
      let offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice
      return offset
    }

    // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
    // five data items: x, y, width, height, and a confidence score. Each grid
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 13x13x125 elements (actually x128 instead of x125 because in
    // Metal the number of channels must be a multiple of 4).

    for cy in 0..<gridHeight {
      for cx in 0..<gridWidth {
        for b in 0..<boxesPerCell {

          // The 13x13x125 image is arranged in planes of 4 channels. First are
          // channels 0-3 for the entire image, then channels 4-7 for the whole
          // image, then channels 8-11, and so on. Since we have 128 channels,
          // there are 128/4 = 32 of these planes (a.k.a. texture slices).
          //
          //    0123 0123 0123 ... 0123    ^
          //    0123 0123 0123 ... 0123    |
          //    0123 0123 0123 ... 0123    13 rows
          //    ...                        |
          //    0123 0123 0123 ... 0123    v
          //    4567 4557 4567 ... 4567
          //    etc
          //    <----- 13 columns ---->
          //
          // For the first bounding box (b=0) we have to read channels 0-24,
          // for b=1 we have to read channels 25-49, and so on. Unfortunately,
          // these 25 channels are spread out over multiple slices. We use a
          // helper function to find the correct place in the features array.
          // (Note: It might be quicker / more convenient to transpose this
          // array so that all 125 channels are stored consecutively instead
          // of being scattered over multiple texture slices.)
          let channel = b*(numClasses + 5)
          let tx = features[offset(channel, cx, cy)]
          let ty = features[offset(channel + 1, cx, cy)]
          let tw = features[offset(channel + 2, cx, cy)]
          let th = features[offset(channel + 3, cx, cy)]
          let tc = features[offset(channel + 4, cx, cy)]

          // The predicted tx and ty coordinates are relative to the location
          // of the grid cell; we use the logistic sigmoid to constrain these
          // coordinates to the range 0 - 1. Then we add the cell coordinates
          // (0-12) and multiply by the number of pixels per grid cell (32).
          // Now x and y represent center of the bounding box in the original
          // 416x416 image space.
          let x = (Float(cx) + sigmoid(tx)) * blockSize
          let y = (Float(cy) + sigmoid(ty)) * blockSize

          // The size of the bounding box, tw and th, is predicted relative to
          // the size of an "anchor" box. Here we also transform the width and
          // height into the original 416x416 image space.
          let w = exp(tw) * anchors[2*b    ] * blockSize
          let h = exp(th) * anchors[2*b + 1] * blockSize

          // The confidence value for the bounding box is given by tc. We use
          // the logistic sigmoid to turn this into a percentage.
          let confidence = sigmoid(tc)

          // Gather the predicted classes for this anchor box and softmax them,
          // so we can interpret these numbers as percentages.
          var classes = [Float](repeating: 0, count: numClasses)
          for c in 0..<numClasses {
            classes[c] = features[offset(channel + 5 + c, cx, cy)]
          }
          classes = softmax(classes)

          // Find the index of the class with the largest score.
          let (detectedClass, bestClassScore) = classes.argmax()

          // Combine the confidence score for the bounding box, which tells us
          // how likely it is that there is an object in this box (but not what
          // kind of object it is), with the largest class prediction, which
          // tells us what kind of object it detected (but not where).
          let confidenceInClass = bestClassScore * confidence

          // Since we compute 13x13x5 = 845 bounding boxes, we only want to
          // keep the ones whose combined score is over a certain threshold.
          if confidenceInClass > 0.3 {
            let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                              width: CGFloat(w), height: CGFloat(h))

            let prediction = Prediction(classIndex: detectedClass,
                                        score: confidenceInClass,
                                        rect: rect)
            predictions.append(prediction)
          }
        }
      }
    }

    // We already filtered out any bounding boxes that have very low scores,
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
    return nonMaxSuppression(boxes: predictions, limit: 10, threshold: 0.5)
  }

  // The weights (and bias terms) must be provided by a data source object.
  // This also returns an MPSCNNConvolutionDescriptor that has the kernel size,
  // number of channels, which activation function to use, etc.
  class DataSource: NSObject, MPSCNNConvolutionDataSource {
    let name: String
    let kernelWidth: Int
    let kernelHeight: Int
    let inputFeatureChannels: Int
    let outputFeatureChannels: Int
    let useLeaky: Bool

    var data: Data?

    init(_ name: String, _ kernelWidth: Int, _ kernelHeight: Int,
         _ inputFeatureChannels: Int, _ outputFeatureChannels: Int,
         useLeaky: Bool = true) {
      self.name = name
      self.kernelWidth = kernelWidth
      self.kernelHeight = kernelHeight
      self.inputFeatureChannels = inputFeatureChannels
      self.outputFeatureChannels = outputFeatureChannels
      self.useLeaky = useLeaky
    }

    func descriptor() -> MPSCNNConvolutionDescriptor {
      let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernelWidth,
                                             kernelHeight: kernelHeight,
                                             inputFeatureChannels: inputFeatureChannels,
                                             outputFeatureChannels: outputFeatureChannels)
      if useLeaky {
        desc.neuronType = .reLU
        desc.neuronParameterA = 0.1

        // This layer has batch normalization applied to it. The data for this
        // layer is stored as: [ weights | mean | variance | gamma | beta ].
        data?.withUnsafeBytes { (ptr: UnsafePointer<Float>) -> Void in
          let weightsSize = outputFeatureChannels * kernelHeight * kernelWidth * inputFeatureChannels
          let mean = ptr.advanced(by: weightsSize)
          let variance = mean.advanced(by: outputFeatureChannels)
          let gamma = variance.advanced(by: outputFeatureChannels)
          let beta = gamma.advanced(by: outputFeatureChannels)
          desc.setBatchNormalizationParametersForInferenceWithMean(mean,
                  variance: variance, gamma: gamma, beta: beta, epsilon: 1e-3)
        }
      } else {
        desc.neuronType = .none
      }
      return desc
    }

    func weights() -> UnsafeMutableRawPointer {
      return UnsafeMutableRawPointer(mutating: (data! as NSData).bytes)
    }

    func biasTerms() -> UnsafeMutablePointer<Float>? {
      return nil
    }

    func load() -> Bool {
      if let url = Bundle.main.url(forResource: name, withExtension: "bin") {
        do {
          data = try Data(contentsOf: url)
          return true
        } catch {
          print("Error: could not load \(url): \(error)")
        }
      }
      return false
    }

    func purge() {
      data = nil
    }

    func label() -> String? {
      return name
    }

    func dataType() -> MPSDataType {
      return .float32
    }
  }

  // This class describes the padding we're using on the pool6 layer.
  class Pool6Padding: NSObject, MPSNNPadding {
    override init() {
      super.init()
    }

    static var supportsSecureCoding: Bool = true

    required init?(coder aDecoder: NSCoder) {
      super.init()
    }

    func encode(with aCoder: NSCoder) {
      // nothing to do here
    }

    func paddingMethod() -> MPSNNPaddingMethod {
      return [ .custom, .sizeSame ]
    }

    func destinationImageDescriptor(forSourceImages sourceImages: [MPSImage],
                                    sourceStates: [MPSState]?,
                                    for kernel: MPSKernel,
                                    suggestedDescriptor inDescriptor: MPSImageDescriptor) -> MPSImageDescriptor {
      if let kernel = kernel as? MPSCNNPooling {
        kernel.offset = MPSOffset(x: 1, y: 1, z: 0)
        kernel.edgeMode = .clamp
      }
      return inDescriptor
    }

    func label() -> String {
      return "Pool6Padding"
    }
  }
}
