import UIKit

extension UIImage {
  /** 
    Converts the image into an array of RGBA bytes.
   */
  @nonobjc public func toByteArray() -> [UInt8] {
    let width = Int(size.width)
    let height = Int(size.height)
    var bytes = [UInt8](repeating: 0, count: width * height * 4)

    bytes.withUnsafeMutableBytes { ptr in
      if let context = CGContext(
                    data: ptr.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {

        if let image = self.cgImage {
          let rect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
          context.draw(image, in: rect)
        }
      }
    }
    return bytes
  }

  /**
    Creates a new UIImage from an array of RGBA bytes.
   */
  @nonobjc public class func fromByteArray(_ bytes: UnsafeMutableRawPointer,
                                           width: Int,
                                           height: Int) -> UIImage {

    if let context = CGContext(data: bytes, width: width, height: height,
                               bitsPerComponent: 8, bytesPerRow: width * 4,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
       let cgImage = context.makeImage() {
      return UIImage(cgImage: cgImage, scale: 0, orientation: .up)
    } else {
      return UIImage()
    }
  }
}
