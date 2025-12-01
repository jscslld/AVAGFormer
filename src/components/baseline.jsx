import React from 'react';
import { render } from 'react-dom';
import { FaCreativeCommons } from 'react-icons/fa';
import { Table, Typography } from 'antd';
import { Carousel } from 'antd';
import {
  FrownOutlined,
  SmileOutlined,
  SyncOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { Bubble } from '@ant-design/x';
import { Button, Flex, Space, Spin } from 'antd';
import ReactPlayer from 'react-player';
import classNames from 'classnames';

const { Text } = Typography;

export default class Baseline extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-section">
        <h2 className="uk-text-bold uk-heading-line uk-text-center">
          <span>Baseline: AVAGFormer</span>
        </h2>
        <img src="architecture.jpg" className="uk-align-center" alt="" />
        <p style={{ textAlign: 'justify' }}>
          AVAGFormer first extracts audio and visual features using pretrained
          backbones. Visual features are enhanced by a Transformer Encoder and
          FPN to fuse global context and multi-scale spatial details. The
          semantic-conditioned cross-modal mixer then aligns and integrates
          audio and visual features, producing function- and dependency-specific
          representations conditioned on audio cues. Finally, a dual-head
          query-conditioned decoder generates function and dependency affordance
          masks.
        </p>
        <h3>
          <span>Comparison with SOTA Models</span>
        </h3>
        <img src="sota.jpg" className="uk-align-center" alt="" />
      </div>
    );
  }
}
