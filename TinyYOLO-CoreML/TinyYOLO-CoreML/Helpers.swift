import Foundation
import UIKit
import CoreML
import Accelerate

let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

/*
  The tiny-yolo-voc network from YOLOv2. https://pjreddie.com/darknet/yolo/

  This implementation is cobbled together from the following sources:

  - https://github.com/pjreddie/darknet
  - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java
  - https://github.com/allanzelener/YAD2K
*/
class YOLO {
  public static let inputWidth = 416
  public static let inputHeight = 416

  public struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }

  let model = TinyYOLO()

  public init() { }

  public func predict(image: CVPixelBuffer) throws -> [Prediction] {
    if let output = try? model.prediction(image: image) {
      return computeBoundingBoxes(features: output.grid)
    } else {
      return []
    }
  }

  public func computeBoundingBoxes(features: MLMultiArray) -> [Prediction] {
    assert(features.count == 125*13*13)

    var predictions = [Prediction]()

    let blockSize: Float = 32
    let gridHeight = 13
    let gridWidth = 13
    let boxesPerCell = 5
    let numClasses = 20

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

          // For the first bounding box (b=0) we have to read channels 0-24,
          // for b=1 we have to read channels 25-49, and so on.
          let channel = b*(numClasses + 5)
          let tx = features[[channel    , cx, cy] as [NSNumber]].floatValue
          let ty = features[[channel + 1, cx, cy] as [NSNumber]].floatValue
          let tw = features[[channel + 2, cx, cy] as [NSNumber]].floatValue
          let th = features[[channel + 3, cx, cy] as [NSNumber]].floatValue
          let tc = features[[channel + 4, cx, cy] as [NSNumber]].floatValue

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
            classes[c] = features[[channel + 5 + c, cx, cy] as [NSNumber]].floatValue
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
}

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.

  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Parameters:
    - boxes: an array of bounding boxes and their scores
    - limit: the maximum number of boxes that will be selected
    - threshold: used to decide whether boxes overlap too much
*/
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count

  // The algorithm is simple: Start with the box that has the highest score.
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      selected.append(boxA)
      if selected.count >= limit { break }

      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  return selected
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(a: CGRect, b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
  return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
  /**
    Returns the index and value of the largest element in the array.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count {
      if self[i] > maxValue {
        maxValue = self[i]
        maxIndex = i
      }
    }
    return (maxIndex, maxValue)
  }
}

/**
  Logistic sigmoid.
*/
public func sigmoid(_ x: Float) -> Float {
  return 1 / (1 + exp(-x))
}

/**
  Computes the "softmax" function over an array.

  Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

  This is what softmax looks like in "pseudocode" (actually using Python
  and numpy):

      x -= np.max(x)
      exp_scores = np.exp(x)
      softmax = exp_scores / np.sum(exp_scores)

  First we shift the values of x so that the highest value in the array is 0.
  This ensures numerical stability with the exponents, so they don't blow up.
*/
public func softmax(_ x: [Float]) -> [Float] {
  var x = x
  let len = vDSP_Length(x.count)

  // Find the maximum value in the input array.
  var max: Float = 0
  vDSP_maxv(x, 1, &max, len)

  // Subtract the maximum from all the elements in the array.
  // Now the highest value in the array is 0.
  max = -max
  vDSP_vsadd(x, 1, &max, &x, 1, len)

  // Exponentiate all the elements in the array.
  var count = Int32(x.count)
  vvexpf(&x, x, &count)

  // Compute the sum of all exponentiated values.
  var sum: Float = 0
  vDSP_sve(x, 1, &sum, len)

  // Divide each element by the sum. This normalizes the array contents
  // so that they all add up to 1.
  vDSP_vsdiv(x, 1, &sum, &x, 1, len)

  return x
}
