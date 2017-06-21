import Foundation
import Accelerate

/* Utility functions for dealing with 16-bit floating point values in Swift. */

/**
  Since Swift has no datatype for a 16-bit float we use `UInt16`s instead,
  which take up the same amount of memory. (Note: The simd framework does 
  have "half" types but only for 2, 3, or 4-element vectors, not scalars.)
*/
public typealias Float16 = UInt16

/**
  Creates a new array of Swift `Float` values from a buffer of float-16s.
*/
public func float16to32(_ input: UnsafeMutablePointer<Float16>, count: Int) -> [Float] {
  var output = [Float](repeating: 0, count: count)
  float16to32(input: input, output: &output, count: count)
  return output
}

/**
  Converts a buffer of float-16s into a buffer of `Float`s, in-place.
*/
public func float16to32(input: UnsafeMutablePointer<Float16>, output: UnsafeMutableRawPointer, count: Int) {
  var bufferFloat16 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 2)
  var bufferFloat32 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 4)

  if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
    print("Error converting float16 to float32")
  }
}

/**
  Creates a new array of float-16 values from a buffer of `Float`s.
*/
public func float32to16(_ input: UnsafeMutablePointer<Float>, count: Int) -> [Float16] {
  var output = [Float16](repeating: 0, count: count)
  float32to16(input: input, output: &output, count: count)
  return output
}

/**
  Converts a buffer of `Float`s into a buffer of float-16s, in-place.
*/
public func float32to16(input: UnsafeMutablePointer<Float>, output: UnsafeMutableRawPointer, count: Int) {
  var bufferFloat32 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 4)
  var bufferFloat16 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 2)

  if vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0) != kvImageNoError {
    print("Error converting float32 to float16")
  }
}
