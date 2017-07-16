import UIKit
import Metal
import AVFoundation
import CoreMedia

class ViewController: UIViewController {
  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var timeLabel: UILabel!
  @IBOutlet weak var debugImageView: UIImageView!

  var device: MTLDevice!
  var videoCapture: VideoCapture!
  var commandQueue: MTLCommandQueue!

  var yolo: YOLO!
  var boundingBoxes = [BoundingBox]()
  var colors: [UIColor] = []
    
  var framesDone = 0
  var frameCapturingStartTime: CFTimeInterval = 0
  let semaphore = DispatchSemaphore(value: 2)

  override func viewDidLoad() {
    super.viewDidLoad()

    timeLabel.text = ""

    device = MTLCreateSystemDefaultDevice()
    if device == nil {
      print("Error: this device does not support Metal")
      return
    }

    commandQueue = device.makeCommandQueue()

    yolo = YOLO(commandQueue: commandQueue)

    setUpBoundingBoxes()
    setUpCamera()
    
    frameCapturingStartTime = CACurrentMediaTime()
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  // MARK: - Initialization

  func setUpBoundingBoxes() {
    for _ in 0..<YOLO.maxBoundingBoxes {
      boundingBoxes.append(BoundingBox())
    }

    // Make colors for the bounding boxes. There is one color for each class,
    // 20 classes in total.
    for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
      for g: CGFloat in [0.3, 0.7] {
        for b: CGFloat in [0.4, 0.8] {
          let color = UIColor(red: r, green: g, blue: b, alpha: 1)
          colors.append(color)
        }
      }
    }
  }

  func setUpCamera() {
    videoCapture = VideoCapture(device: device)
    videoCapture.delegate = self
    videoCapture.fps = 50
    videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.vga640x480) { success in
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

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = videoPreview.bounds
  }

  // MARK: - Doing inference

  func predict(texture: MTLTexture) {
    yolo.predict(texture: texture) { result in
      DispatchQueue.main.async {
        self.show(predictions: result.predictions)

        if let texture = result.debugTexture {
          self.debugImageView.image = UIImage.image(texture: texture)
        }

        let fps = self.measureFPS()
        self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", result.elapsed, fps)

        self.semaphore.signal()
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
    for i in 0..<boundingBoxes.count {
      if i < predictions.count {
        let prediction = predictions[i]

        // The predicted bounding box is in the coordinate space of the input
        // image, which is a square image of 416x416 pixels. We want to show it
        // on the video preview, which is as wide as the screen and has a 4:3
        // aspect ratio. The video preview also may be letterboxed at the top
        // and bottom.
        let width = view.bounds.width
        let height = width * 4 / 3
        let scaleX = width / 416
        let scaleY = height / 416
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

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
    // For debugging.
    //predict(texture: loadTexture(named: "dog416.png")!); return

    // The semaphore is necessary because the call to predict() does not block.
    // If we _would_ be blocking, then AVCapture will automatically drop frames
    // that come in while we're still busy. But since we don't block, all these
    // new frames get scheduled to run in the future and we end up with a backlog
    // of unprocessed frames. So we're using the semaphore to block if predict()
    // is already processing 2 frames, and we wait until the first of those is
    // done. Any new frames that come in during that time will simply be dropped.
    semaphore.wait()

    if let texture = texture {
      predict(texture: texture)
    }
  }
}
