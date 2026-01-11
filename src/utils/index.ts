import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';
import isEqual from 'react-fast-compare';

import {AgentSacConstructorProps} from '../@types';
import {AgentSac, AgentSacTrainable} from '../classes';
import {VERSION} from '../constants';

// https://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
export const getRandomInt = (min: number, max: number)  => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const assertNumericArray = (e: unknown): number[] => {
  assert(Array.isArray(e));

  return e.map(i => {
    assert(typeof i === 'number');
    return i;
  });
}

export const assertShape = (tensor: tf.Tensor, shape: readonly number[]) =>
  assert(isEqual(tensor.shape, shape));

export const assertScalar = (t: tf.Tensor): tf.Scalar => {
  assert(t.rank === 0);
  return t as tf.Scalar;
};

export const getTrainableOnlyWeights = (layersModel: tf.LayersModel) =>
  layersModel
    .getWeights(true /* trainableOnly */)
    .map(w => {
      assert(w instanceof tf.Variable);
      return w;
    });

const getModelKey = (modelName: string) => `indexeddb://${modelName}-${VERSION}`;

export const saveModel = (
  model: tf.LayersModel
) => model.save(getModelKey(model.name));

export const loadModelByName = async (
  modelName: string
): Promise<tf.LayersModel | null> => {
  const key = getModelKey(modelName);
  const modelsInfo = await tf.io.listModels();
  if (key in modelsInfo) return tf.loadLayersModel(key);
  return null;
};

export const createConvEncoder = (
  inputs: tf.SymbolicTensor
): tf.SymbolicTensor => {
  const padding = 'valid'
  const kernelInitializer = 'glorotNormal'
  const biasInitializer = 'glorotNormal'

  let outputs = tf.layers.conv2d({
    filters: 16,
    kernelSize: 5,
    strides: 2,
    padding,
    kernelInitializer,
    biasInitializer,
    activation: 'relu',
    trainable: true,
  }).apply(inputs);

  outputs = tf.layers.maxPooling2d({poolSize:2}).apply(outputs);

  outputs = tf.layers.conv2d({
    filters: 16,
    kernelSize: 3,
    strides: 1,
    padding,
    kernelInitializer,
    biasInitializer,
    activation: 'relu',
    trainable: true,
  }).apply(outputs);

  outputs = tf.layers.maxPooling2d({poolSize:2}).apply(outputs);
  outputs = tf.layers.flatten().apply(outputs);

  assert(outputs instanceof tf.SymbolicTensor);
  return outputs
};

export const createAgentSac = async ({
  agentSacProps,
}: {
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
} = Object.create(null)) => {
  const agent = new AgentSac(agentSacProps);
  await agent.initialize();
  return {agent};
};

export const createAgentSacTrainable = async ({
  agentSacProps, 
}: {
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
} = Object.create(null)) => {
  const agent = new AgentSacTrainable(agentSacProps);
  await agent.initialize();
  await agent.checkpoint();
  return {agent};
};