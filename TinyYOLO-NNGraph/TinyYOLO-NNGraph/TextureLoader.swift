import MetalKit

let textureLoader: MTKTextureLoader = {
  return MTKTextureLoader(device: MTLCreateSystemDefaultDevice()!)
}()

/**
  Loads a texture from the main bundle.
*/
public func loadTexture(named filename: String) -> MTLTexture? {
  if let url = Bundle.main.url(forResource: filename, withExtension: "") {
    return loadTexture(url: url)
  } else {
    print("Error: could not find image \(filename)")
    return nil
  }
}

/**
  Loads a texture from the specified URL.
*/
public func loadTexture(url: URL) -> MTLTexture? {
  do {
    return try textureLoader.newTexture(URL: url, options: [
      MTKTextureLoader.Option.SRGB : NSNumber(value: false)
    ])
  } catch {
    print("Error: could not load texture \(error)")
    return nil
  }
}
