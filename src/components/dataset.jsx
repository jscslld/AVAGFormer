import { render } from 'react-dom';
import { FaCreativeCommons } from 'react-icons/fa';
import { Table, Typography } from 'antd';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
const { Text } = Typography;
import React, { useState, useEffect, useRef } from 'react';
import { Carousel } from 'antd';
function ImageMaskSlider({
  imageSrc,
  funcMaskSrc,
  depMaskSrc,
  audioSrc,
  transcript,
  maskOpacity = 0.6,
}) {
  const [sliderValue, setSliderValue] = useState(100);

  const clip = `inset(0 ${100 - sliderValue}% 0 0)`;

  return (
    <div
      style={{
        marginTop: 20,
        marginLeft: 'auto',
        marginRight: 'auto',
        // 或者直接 margin: '20px auto',

        backgroundColor: '#ffffff',
        borderRadius: 10,
        boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
        border: '1px solid #e5e5e5',
        overflow: 'hidden',
        width: '90%',
      }}
    >
      {/* Card Title */}
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid #e5e5e5',
          backgroundColor: '#fafafa',
          fontWeight: '600',
          fontSize: '18px',
        }}
      >
        {transcript || 'Transcript'}
      </div>

      {/* Card Body */}
      <div style={{ padding: 16 }}>
        {/* 上层：左右两栏图像 */}
        <div style={{ display: 'flex', gap: 10 }}>
          {/* 左栏 */}
          <div style={{ flex: 1 }}>
            <div style={{ position: 'relative' }}>
              <img
                src={imageSrc}
                alt="base"
                style={{ height: '512px', display: 'block' }}
              />

              <img
                src={funcMaskSrc}
                alt="function mask"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  height: '512px',
                  // objectFit: 'contain',
                  opacity: maskOpacity,
                  clipPath: clip,
                  pointerEvents: 'none',
                }}
              />
            </div>

            <div
              style={{ textAlign: 'center', marginTop: 6, fontWeight: 'bold' }}
            >
              Function Mask
            </div>
          </div>

          {/* 右栏 */}
          <div style={{ flex: 1 }}>
            <div style={{ position: 'relative' }}>
              <img
                src={imageSrc}
                alt="base"
                style={{ height: '512px', display: 'block' }}
              />

              <img
                src={depMaskSrc}
                alt="dependency mask"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  height: '512px',
                  // objectFit: 'cover',
                  opacity: maskOpacity,
                  clipPath: clip,
                  pointerEvents: 'none',
                }}
              />
            </div>

            <div
              style={{ textAlign: 'center', marginTop: 6, fontWeight: 'bold' }}
            >
              Dependency Mask
            </div>
          </div>
        </div>

        {/* Slider */}
        <div style={{ marginTop: 10 }}>
          <input
            type="range"
            min="0"
            max="100"
            value={sliderValue}
            onChange={(e) => setSliderValue(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        {/* Audio */}
        {audioSrc && (
          <div style={{ marginTop: 15 }}>
            <audio controls src={audioSrc} style={{ width: '100%' }} />
          </div>
        )}
      </div>
    </div>
  );
}

export default class Dataset extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-section">
        <h2 className="uk-text-bold uk-heading-line uk-text-center">
          <span>Dataset: AVAGD</span>
        </h2>
        <img
          src="statistics.jpg"
          className="uk-align-center"
          alt=""
          width={'90%'}
        />
        <p style={{ textAlign: 'justify' }}>
          AVAGD dataset covers 7 domains, 55 affordance categories, and 97
          object categories, encompassing common human–object interactions that
          generate distinctive sounds. Unlike the fixed-length audio in
          AVSBench, AVAGD features variable-duration audio clips posing a
          greater challenge for model perception. The dataset includes 12,768
          annotated images and 5,203 audio clips, making it the first publicly
          available affordance grounding dataset with interaction audio. To
          evaluate zero-shot generalization, the dataset is split into seen and
          unseen subsets. The unseen subset consists of 29 object-affordance
          categories for testing, while the seen subset includes 84 categories,
          where images and audio within each category are further divided into
          80% for training and 20% for validation.
        </p>
        <h3>
          <span>Examples</span>
        </h3>
        <Carousel arrows infinite={false} dots={false}>
          <ImageMaskSlider
            imageSrc="./examples/bike/ride/OIDV7_0a31ecb885a97211.jpg"
            funcMaskSrc="./examples/bike/ride/OIDV7_0a31ecb885a97211_Function_mask.png"
            depMaskSrc="./examples/bike/ride/OIDV7_0a31ecb885a97211_Dependency_mask.png"
            audioSrc="./examples/bike/ride/1zVsl0uqdyE_17.99_31.52.mp3"
            transcript="Bike - Ride"
          ></ImageMaskSlider>
          <ImageMaskSlider
            imageSrc="./examples/axe/chop/Flickr1_82949260_9859ca6c20_o.jpg"
            funcMaskSrc="./examples/axe/chop/Flickr1_82949260_9859ca6c20_o_Function_mask.png"
            depMaskSrc="./examples/axe/chop/Flickr1_82949260_9859ca6c20_o_Dependency_mask.png"
            audioSrc="./examples/axe/chop/0OH2mR4-nX0_2.82_5.72.mp3"
            transcript="Axe - Chop"
          ></ImageMaskSlider>
          <ImageMaskSlider
            imageSrc="./examples/bottle_cap/twist/Products-10K_train-00013-of-00029_88.jpg"
            funcMaskSrc="./examples/bottle_cap/twist/Products-10K_train-00013-of-00029_88_Function_mask.png"
            depMaskSrc="./examples/bottle_cap/twist/Products-10K_train-00013-of-00029_88_Dependency_mask.png"
            audioSrc="./examples/bottle_cap/twist/P1FXGXxfcVw_2.23_4.44.mp3"
            transcript="Bottle Cap - Twist"
          ></ImageMaskSlider>
        </Carousel>
      </div>
    );
  }
}
