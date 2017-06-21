import Metal

extension MTLTexture {
  /**
    Creates a new array of `Float`s and copies the texture's pixels into it.
  */
  public func toFloatArray(width: Int, height: Int, featureChannels: Int) -> [Float] {
    return toArray(width: width, height: height,
                   featureChannels: featureChannels, initial: Float(0))
  }

  /**
    Creates a new array of `Float16`s and copies the texture's pixels into it.
  */
  public func toFloat16Array(width: Int, height: Int, featureChannels: Int) -> [Float16] {
    return toArray(width: width, height: height,
                   featureChannels: featureChannels, initial: Float16(0))
  }

  /**
    Creates a new array of `UInt8`s and copies the texture's pixels into it.
  */
  public func toUInt8Array(width: Int, height: Int, featureChannels: Int) -> [UInt8] {
    return toArray(width: width, height: height,
                   featureChannels: featureChannels, initial: UInt8(0))
  }

  /**
    Convenience function that copies the texture's pixel data to a Swift array.

    The type of `initial` determines the type of the output array. In the
    following example, the type of bytes is `[UInt8]`.

        let bytes = texture.toArray(width: 100, height: 100, featureChannels: 4, initial: UInt8(0))

    - Parameters:
      - featureChannels: The number of color components per pixel: must be 1, 2, or 4.
      - initial: This parameter is necessary because we need to give the array 
        an initial value. Unfortunately, we can't do `[T](repeating: T(0), ...)` 
        since `T` could be anything and may not have an init that takes a literal 
        value.
  */
  func toArray<T>(width: Int, height: Int, featureChannels: Int, initial: T) -> [T] {
    assert(featureChannels != 3 && featureChannels <= 4, "channels must be 1, 2, or 4")

    var bytes = [T](repeating: initial, count: width * height * featureChannels)
    let region = MTLRegionMake2D(0, 0, width, height)
    getBytes(&bytes, bytesPerRow: width * featureChannels * MemoryLayout<T>.stride,
             from: region, mipmapLevel: 0)
    return bytes
  }
}
