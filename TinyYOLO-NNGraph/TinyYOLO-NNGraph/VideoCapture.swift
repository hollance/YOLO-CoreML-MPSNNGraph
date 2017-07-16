import UIKit
import AVFoundation
import CoreVideo
import Metal

public protocol VideoCaptureDelegate: class {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime)
}

public class VideoCapture: NSObject {
  public var previewLayer: AVCaptureVideoPreviewLayer?
  public weak var delegate: VideoCaptureDelegate?
  public var fps = 15

  let device: MTLDevice
  var textureCache: CVMetalTextureCache?
  let captureSession = AVCaptureSession()
  let videoOutput = AVCaptureVideoDataOutput()
  let queue = DispatchQueue(label: "net.machinethink.camera-queue")

  var lastTimestamp = CMTime()

  public init(device: MTLDevice) {
    self.device = device
    super.init()
  }

  public func setUp(sessionPreset: AVCaptureSession.Preset = .medium,
                    completion: @escaping (Bool) -> Void) {
    queue.async {
      let success = self.setUpCamera(sessionPreset: sessionPreset)
      DispatchQueue.main.async {
        completion(success)
      }
    }
  }

  func setUpCamera(sessionPreset: AVCaptureSession.Preset) -> Bool {
    guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache) == kCVReturnSuccess else {
      print("Error: could not create a texture cache")
      return false
    }

    captureSession.beginConfiguration()
    captureSession.sessionPreset = sessionPreset

    guard let captureDevice = AVCaptureDevice.default(for: AVMediaType.video) else {
      print("Error: no video devices available")
      return false
    }

    guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
      print("Error: could not create AVCaptureDeviceInput")
      return false
    }

    if captureSession.canAddInput(videoInput) {
      captureSession.addInput(videoInput)
    }

    let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    previewLayer.videoGravity = AVLayerVideoGravity.resizeAspect
    previewLayer.connection?.videoOrientation = .portrait
    self.previewLayer = previewLayer

    let settings: [String : Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]

    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: queue)
    if captureSession.canAddOutput(videoOutput) {
      captureSession.addOutput(videoOutput)
    }

    // We want the buffers to be in portrait orientation otherwise they are
    // rotated by 90 degrees. Need to set this _after_ addOutput()!
    videoOutput.connection(with: AVMediaType.video)?.videoOrientation = .portrait

    captureSession.commitConfiguration()
    return true
  }

  public func start() {
    if !captureSession.isRunning {
      captureSession.startRunning()
    }
  }

  public func stop() {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }
  }

  func convertToMTLTexture(sampleBuffer: CMSampleBuffer?) -> MTLTexture? {
    if let textureCache = textureCache,
       let sampleBuffer = sampleBuffer,
       let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

      let width = CVPixelBufferGetWidth(imageBuffer)
      let height = CVPixelBufferGetHeight(imageBuffer)

      var texture: CVMetalTexture?
      CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
          imageBuffer, nil, .bgra8Unorm, width, height, 0, &texture)

      if let texture = texture {
        return CVMetalTextureGetTexture(texture)
      }
    }
    return nil
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    // Because lowering the capture device's FPS looks ugly in the preview,
    // we capture at full speed but only call the delegate at its desired
    // framerate.
    let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
    let deltaTime = timestamp - lastTimestamp
    if deltaTime >= CMTimeMake(1, Int32(fps)) {
      lastTimestamp = timestamp
      let texture = convertToMTLTexture(sampleBuffer: sampleBuffer)
      delegate?.videoCapture(self, didCaptureVideoTexture: texture, timestamp: timestamp)
    }
  }

  public func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    //print("dropped frame")
  }
}
