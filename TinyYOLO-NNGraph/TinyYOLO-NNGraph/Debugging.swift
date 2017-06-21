import Foundation
import MetalPerformanceShaders

/**
  Diagnostic tool for verifying that the neural network works correctly:
  prints out the channels for a given pixel coordinate.

  Writing `printChannelsForPixel(x: 5, y: 10, ...)` is the same as doing
  `print(layer_output[0, 10, 5, :])` in Python with layer output from Keras.
  Note that x and y are swapped in the Python code!

  To make sure the layer computes the right thing, feed the exact same image
  through Metal and Keras and compare the layer outputs.
*/
public func printChannelsForPixel(x: Int, y: Int, image: MPSImage) {
  let layerOutput = image.toFloatArray()
  print("Total size: \(layerOutput.count) floats")
  let w = image.width
  let h = image.height
  let s = (image.featureChannels + 3)/4
  for b in 0..<image.numberOfImages {
    for i in 0..<s {
      print("[batch index \(b), slice \(i) of \(s)]")
      for j in 0..<4 {
        print(layerOutput[b*s*h*w*4 + i*h*w*4 + y*w*4 + x*4 + j])
      }
    }
  }
}
