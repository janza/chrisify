package facefinder

import (
	"bytes"
	"image"

	"gocv.io/x/gocv"
)

// var faceCascade *opencv.HaarCascade

type Finder struct {
	classifier *gocv.CascadeClassifier
}

func NewFinder(xml string) *Finder {
	classifier := gocv.NewCascadeClassifier()
	classifier.Load(xml)
	return &Finder{
		classifier: &classifier,
	}
}

func (f *Finder) Detect(imagePath string) (image.Image, []image.Rectangle, error) {

	readImage := gocv.IMRead(imagePath, gocv.IMReadColor)

	faces := f.classifier.DetectMultiScale(readImage)

	encodedImg, err := gocv.IMEncode(".jpg", readImage)
	if err != nil {
		return nil, nil, err
	}

	image, _, err := image.Decode(bytes.NewReader(encodedImg))
	if err != nil {
		return nil, nil, err
	}

	return image, faces, nil
}

func (f *Finder) Close() error {
	return f.classifier.Close()
}
