import UIKit
import Vision
import AVFoundation
import CoreMedia

class ViewController: UIViewController {
    
    //MARK: outlets
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var timeLabel: UILabel!
    @IBOutlet weak var debugImageView: UIImageView!
    
    // true: use Vision to drive Core ML, false: use plain Core ML
    let useVision = false
    
    // Disable this to see the energy impact of just running the neural net,
    // otherwise it also counts the GPU activity of drawing the bounding boxes.
    let drawBoundingBoxes = true
    
    // How many predictions we can do concurrently.
    static let maxInflightBuffers = 3
    
    //creates an instance of the functions that controll the bahavior of the YOLO nural network
    let yolo = YOLO()
    
    //MARK: Private variables
    var videoCapture: VideoCapture!
    var requests = [VNCoreMLRequest]()
    var startTimes: [CFTimeInterval] = []
    
    //creates an array of bounding boxes of differnt colors
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []
    
    //creates an instance of a structure which can store a static image for processing
    let ciContext = CIContext()
    var resizedPixelBuffers: [CVPixelBuffer?] = []
    
    var framesDone = 0
    var frameCapturingStartTime = CACurrentMediaTime()
    
    var inflightBuffer = 0
    //sets up semafores which let us perform multithreading each semafore can be a concurent process
    let semaphore = DispatchSemaphore(value: ViewController.maxInflightBuffers)
    
    //MARK: View did load()
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //sets the time to recognise text to an empty value
        timeLabel.text = ""
        
        //initalize classes
        setUpBoundingBoxes()
        setUpCoreImage()
        setUpVision()
        setUpCamera()
        
        //start the frame capturing
        frameCapturingStartTime = CACurrentMediaTime()
    }
    
    //error parsing for memory warnings
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }
    
    // MARK: - Initialization
    
    func setUpBoundingBoxes() {
        //add bounding boxes equal to the maximum number of bounding boxes
        for _ in 0..<YOLO.maxBoundingBoxes {
            boundingBoxes.append(BoundingBox())
        }
        
        // Make colors for the bounding boxes. There is one color for each class,
        // 20 classes in total.
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g: CGFloat in [0.3, 0.7] {
                for b: CGFloat in [0.4, 0.8] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    //addd the colors to the array of colors created before
                    colors.append(color)
                }
            }
        }
    }
    
    func setUpCoreImage() {
        // Since we might be running several requests in parallel, we also need
        // to do the resizing in different pixel buffers or we might overwrite a
        // pixel buffer that's already in use.
        // one pixel buffer will hold one image section which we are going to be checking. thius one pixel buffer is an 'image' or image section representing one object
        for _ in 0..<YOLO.maxBoundingBoxes {
            //CVPixel Buffer is an image stored as pixels
            var resizedPixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(nil, YOLO.inputWidth, YOLO.inputHeight,
                                             kCVPixelFormatType_32BGRA, nil,
                                             &resizedPixelBuffer)
            //error handeling
            if status != kCVReturnSuccess {
                print("Error: could not create resized pixel buffer", status)
            }
            //add the pixel buffer/image
            resizedPixelBuffers.append(resizedPixelBuffer)
        }
    }
    
    func setUpVision() {
        //attempts to create the yolo neural network
        guard let visionModel = try? VNCoreMLModel(for: yolo.model.model) else {
            print("Error: could not create Vision model")
            return
        }
        
        
        for _ in 0..<ViewController.maxInflightBuffers {
            //sets up a bunch of requests to send to the neural network
            let request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            
            // NOTE: If you choose another crop/scale option, then you must also
            // change how the BoundingBox objects get scaled when they are drawn.
            // Currently they assume the full input image is used.
            request.imageCropAndScaleOption = .scaleFill
            requests.append(request)
        }
    }
    
    
    func setUpCamera() {
        //initalizes a video caputre
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.desiredFrameRate = 240
        videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.hd1280x720) { success in
            if success {
                // Add the video preview into the UI.
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                
                // Add the bounding box layers to the UI, on top of the video preview.
                for box in self.boundingBoxes {
                    box.addToLayer(self.videoPreview.layer)
                }
                
                // Once everything is set up, we can start capturing live video.
                self.videoCapture.start()
            }
        }
    }
    
    // MARK: - UI stuff
    
    //arrainges views on the screen
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    //sets the status bar style
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    //sets the video from the camera to take up all of its allocated space
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
    
    // MARK: - Doing inference
    
    //attempts to find an sub image in the greater image
    func predict(image: UIImage) {
        if let pixelBuffer = image.pixelBuffer(width: YOLO.inputWidth, height: YOLO.inputHeight) {
            //fills a pixel buffer with a sub image
            predict(pixelBuffer: pixelBuffer, inflightIndex: 0)
        }
    }
    
    
    func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
        // Measure how long it takes to predict a single video frame.
        let startTime = CACurrentMediaTime()
        
        // This is an alternative way to resize the image (using vImage):
        //if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
        //                                              width: YOLO.inputWidth,
        //                                              height: YOLO.inputHeight) {
        
        // Resize the input with Core Image to 416x416.
        if let resizedPixelBuffer = resizedPixelBuffers[inflightIndex] {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let sx = CGFloat(YOLO.inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let sy = CGFloat(YOLO.inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
            let scaledImage = ciImage.transformed(by: scaleTransform)
            ciContext.render(scaledImage, to: resizedPixelBuffer)
            
            
            // Give the resized input to our model.
            if let result = try? yolo.predict(image: resizedPixelBuffer),
                let boundingBoxes = result {
                let elapsed = CACurrentMediaTime() - startTime
                showOnMainThread(boundingBoxes, elapsed)
            } else {
                //if the model could not find anything
                print("BOGUS")
            }
        }
        //send the message that this thread has compleeted
        self.semaphore.signal()
    }
    
    func predictUsingVision(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
        // Measure how long it takes to predict a single video frame. Note that
        // predict() can be called on the next frame while the previous one is
        // still being processed. Hence the need to queue up the start times.
        startTimes.append(CACurrentMediaTime())
        
        // Vision will automatically resize the input image.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        let request = requests[inflightIndex]
        
        // Because perform() will block until after the request completes, we
        // run it on a concurrent background queue, so that the next frame can
        // be scheduled in parallel with this one.
        DispatchQueue.global().async {
            try? handler.perform([request])
        }
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        //make an observation and find fproper features
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let features = observations.first?.featureValue.multiArrayValue {
            
            //set the bounding boxes of the features found
            let boundingBoxes = yolo.computeBoundingBoxes(features: features)
            //stop the timer
            let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
            showOnMainThread(boundingBoxes, elapsed)
        } else {
            print("BOGUS!")
        }
        //send compleeted signal
        self.semaphore.signal()
    }
    
    func showOnMainThread(_ boundingBoxes: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
        //redraw the bounding boxes
        if drawBoundingBoxes {
            DispatchQueue.main.async {
                // For debugging, to make sure the resized CVPixelBuffer is correct.
                //var debugImage: CGImage?
                //VTCreateCGImageFromCVPixelBuffer(resizedPixelBuffer, nil, &debugImage)
                //self.debugImageView.image = UIImage(cgImage: debugImage!)
                
                //show the predictions in the view
                self.show(predictions: boundingBoxes)
                //measure the fps
                let fps = self.measureFPS()
                //print the elaped time
                self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)
            }
        }
    }
    
    func measureFPS() -> Double {
        // Measure how many frames were actually delivered per second.
        framesDone += 1
        let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
        let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
        if frameCapturingElapsed > 1 {
            framesDone = 0
            frameCapturingStartTime = CACurrentMediaTime()
        }
        return currentFPSDelivered
    }
    
    
    func show(predictions: [YOLO.Prediction]) {
        //iterate through all of the bounding boxes
        for i in 0..<boundingBoxes.count {
            if i < predictions.count {
                let prediction = predictions[i]
                
                // The predicted bounding box is in the coordinate space of the input
                // image, which is a square image of 416x416 pixels. We want to show it
                // on the video preview, which is as wide as the screen and has a 16:9
                // aspect ratio. The video preview also may be letterboxed at the top
                // and bottom.
                let width = view.bounds.width
                let height = width * 16 / 9
                let scaleX = width / CGFloat(YOLO.inputWidth)
                let scaleY = height / CGFloat(YOLO.inputHeight)
                let top = (view.bounds.height - height) / 2
                
                // Translate and scale the rectangle to our own coordinate system.
                var rect = prediction.rect
                rect.origin.x *= scaleX
                rect.origin.y *= scaleY
                rect.origin.y += top
                rect.size.width *= scaleX
                rect.size.height *= scaleY
                
                // Show the bounding box.
                let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
                let color = colors[prediction.classIndex]
                boundingBoxes[i].show(frame: rect, label: label, color: color)
            } else {
                boundingBoxes[i].hide()
            }
        }
    }
}

//plays the camera's video
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // For debugging.
        //predict(image: UIImage(named: "dog416")!); return
        
        if let pixelBuffer = pixelBuffer {
            // The semaphore will block the capture queue and drop frames when
            // Core ML can't keep up with the camera.
            semaphore.wait()
            
            // For better throughput, we want to schedule multiple prediction requests
            // in parallel. These need to be separate instances, and inflightBuffer is
            // the index of the current request.
            let inflightIndex = inflightBuffer
            inflightBuffer += 1
            if inflightBuffer >= ViewController.maxInflightBuffers {
                inflightBuffer = 0
            }
            
            if useVision {
                // This method should always be called from the same thread!
                // Ain't nobody likes race conditions and crashes.
                self.predictUsingVision(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
            } else {
                // For better throughput, perform the prediction on a concurrent
                // background queue instead of on the serial VideoCapture queue.
                DispatchQueue.global().async {
                    self.predict(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
                }
            }
        }
    }
}
