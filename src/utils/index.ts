import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';
import isEqual from 'react-fast-compare';
import { AgentSacProps } from '../@types';
import { AgentSac } from '../classes';

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

export const createAgentSac = async ({
  agentSacProps = Object.create(null),
}: {
  readonly agentSacProps?: Partial<AgentSacProps>;
}) => {
  const agent = new AgentSac(agentSacProps);
  await agent.init();
  return {agent};
};