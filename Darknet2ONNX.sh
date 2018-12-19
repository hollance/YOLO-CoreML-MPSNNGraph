#!/bin/bash
rm -rf Convert/yad2k/model_data/tiny-yolo-voc.h5
rm -rf TinyYOLO-CoreML/TinyYOLO-CoreML/TinyYOLO.mlmodel

echo "Converting..."
WRONG_FLAG=1
cd Convert/yad2k
if [ ! -f "$1"] || [ ! -f "$2"] || [ -n "$1"] || [ -n "$2"]; then
        echo "$1 not found! Use default config file tiny-yolo-voc.cfg"
	echo "$2 not found! Use default weights file tiny-yolo-voc.weights"
	"python3" yad2k.py -p ../tiny-yolo-voc.cfg ../tiny-yolo-voc.weights model_data/tiny-yolo-voc.h5 
	echo "Generated Keras h5 file."	
else
	chmod +x $1
	chmod +x $2
	echo "Convert $1 and $2."
	"python3" yad2k.py -p $1 $2 model_data/tiny-yolo-voc.h5
fi
if [ $WRONG_FLAG ]; then
	cd ..
	"python" coreml.py 
	echo "Generated Coreml file."
fi
if [ $WRONG_FLAG ]; then
	cd ../TinyYOLO-CoreML/TinyYOLO-CoreML/
	"python" coreml2onnx 
	echo "Generated onnx file $pwd" 
fi
