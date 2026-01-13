import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';

import {Transition} from './@types';
import {ReplyBuffer} from './classes';
import {
  assertNumericArray,
  createAgentSacTrainable,
} from './utils';

const DISABLED = false
const BATCH_SIZE_AMPLIFIER = 10;

void (async () => {
  const batchSize = 100;

  const {agent, checkpoint} = await createAgentSacTrainable({
    agentSacProps: {batchSize},
  });

  const actor = agent.actor;
  assert(actor);

  // eslint-disable-next-line no-restricted-globals
  self.postMessage({
    weights: await Promise.all(actor.getWeights().map(w => w.array())),
  });

  const rb = new ReplyBuffer(
     5000 * batchSize,
    ({state: [telemetry, frameL, frameR], action, reward}: Omit<Transition, 'nextState'>) => {
      frameL.dispose();
      frameR.dispose();
      telemetry.dispose();
      action.dispose();
      reward.dispose();
    },
  );

  const executeSamples = async () => {
    const samples = rb.sample(agent._batchSize) // time fast
    assert(samples.length === agent._batchSize); 
    
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

      const trainingInput: Omit<Transition, 'id' | 'priority'> = {
        state: [tf.stack(telemetries), tf.stack(framesL), tf.stack(framesR)],
        action: tf.stack(actions), 
        reward: tf.stack(rewards), 
        nextState: [
          tf.stack(nextTelemetries),
          tf.stack(nextFramesL),
          tf.stack(nextFramesR),
        ],
      };

      agent.train(trainingInput);
    }); 

    // eslint-disable-next-line no-restricted-globals
    self.postMessage({
      weights: await Promise.all(actor.getWeights().map(w => w.array())),
    });
  };

  void (async () => {
    while (true) {
      await new Promise(resolve => requestAnimationFrame(resolve));

      if (rb.size < agent._batchSize * BATCH_SIZE_AMPLIFIER) continue;

      console.log('Training...');
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
            tf.tensor3d(frameL, agent._frameStackShape!),
            tf.tensor3d(frameR, agent._frameStackShape!)
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
    
        if (i % (BATCH_SIZE_AMPLIFIER * batchSize) === 0) {
          console.log('doing checkpoint');
          void checkpoint() // timer ~ 500ms, don't await intentionally
        }
    })
})()