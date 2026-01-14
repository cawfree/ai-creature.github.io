import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';

import {Transition} from './@types';
import {ReplyBuffer} from './classes';
import {
  assertNumericArray,
} from './utils';

import {createCreatureAgentSacTrainableInstance} from './examples/creature/utils';

const DISABLED = false
const BATCH_SIZE_AMPLIFIER = 10;

void (async () => {

  const agentSacTrainableInstance = await createCreatureAgentSacTrainableInstance({
    agentSacProps: {
      batchSize: 100,
    },
  });

  // eslint-disable-next-line no-restricted-globals
  self.postMessage({
    weights: await Promise.all(agentSacTrainableInstance.actor.getWeights().map(w => w.array())),
  });

  const rb = new ReplyBuffer(
     5000 * agentSacTrainableInstance.batchSize,
    ({state: [telemetry, frameL, frameR], action, reward}: Omit<Transition, 'nextState'>) => {
      frameL.dispose();
      frameR.dispose();
      telemetry.dispose();
      action.dispose();
      reward.dispose();
    },
  );

  const executeSamples = async () => {
    const samples = rb.sample(agentSacTrainableInstance.batchSize) // time fast
    assert(samples.length === agentSacTrainableInstance.batchSize); 
    
    tf.tidy(() => {
      const framesL: tf.Tensor[] = [];
      const framesR: tf.Tensor[] = [];
      const telemetries: tf.Tensor[] = [];
      const actions: tf.Tensor[] = [];
      const rewards: tf.Tensor[] = [];
      const nextFramesL: tf.Tensor[] = [];
      const nextFramesR: tf.Tensor[] = [];
      const nextTelemetries: tf.Tensor[] = [];
    
      for (const {
        state: [telemetry, frameL, frameR], 
        action, 
        reward, 
        nextState: [nextTelemetry, nextFrameL, nextFrameR] 
      } of samples) {
        framesL.push(frameL);
        framesR.push(frameR);
        telemetries.push(telemetry);
        actions.push(action);
        rewards.push(reward);
        nextFramesL.push(nextFrameL);
        nextFramesR.push(nextFrameR);
        nextTelemetries.push(nextTelemetry);
      }

      void agentSacTrainableInstance.train({
        transition: {
          state: [
            tf.stack(telemetries),
            tf.stack(framesL),
            tf.stack(framesR),
          ],
          action: tf.stack(actions), 
          reward: tf.stack(rewards), 
          nextState: [
            tf.stack(nextTelemetries),
            tf.stack(nextFramesL),
            tf.stack(nextFramesR),
          ],
        },
      });
    }); 

    // eslint-disable-next-line no-restricted-globals
    self.postMessage({
      weights: await Promise.all(agentSacTrainableInstance.actor.getWeights().map(w => w.array())),
    });
  };

  void (async () => {
    while (true) {
      await new Promise(resolve => requestAnimationFrame(resolve));

      if (rb.size < agentSacTrainableInstance.batchSize * BATCH_SIZE_AMPLIFIER) continue;

      await executeSamples();
      await new Promise(resolve => setTimeout(resolve, 240));
    }
  })();
    
    /**
     * Decode transition from the main thread.
     * 
     * @param {{ id, state, action, reward }} transition 
     * @returns 
     */
    const decodeTransition = (transition: Record<string, unknown>): Omit<Transition, 'nextState'> => {
        const {id, state, action, reward, priority} = transition

        assert(typeof id === 'number');
        assert(typeof reward === 'number');

        assert(priority === undefined || typeof priority === 'number');

        assert(Array.isArray(state));
        assert(Array.isArray(action)); // [number, number, number]

        const [telemetry, frameL, frameR] = state;
    
        return tf.tidy((): Omit<Transition, 'nextState'> => ({
          id,
          state: [
            tf.tensor1d(telemetry),
            tf.tensor3d(frameL, agentSacTrainableInstance.frameStackShape!),
            tf.tensor3d(frameR, agentSacTrainableInstance.frameStackShape!)
          ],
          action: tf.tensor1d(assertNumericArray(action)),
          reward: tf.tensor1d([reward]),
          priority,
        }));
    }
    
    let i = 0

    // eslint-disable-next-line no-restricted-globals
    self.addEventListener('message', async e => {
        i++

        if (DISABLED) return
        if (i%50 === 0) console.log('RBSIZE: ', rb.size)
    
        switch (e.data.action) {
            case 'newTransition':
                const transition = decodeTransition(e.data.transition)
                rb.add(transition)
                break
            default:
                console.warn('Unknown action')
                break
        }
    
        if (i % (BATCH_SIZE_AMPLIFIER * agentSacTrainableInstance.batchSize) === 0)
          void agentSacTrainableInstance.checkpoint();
    })
})()