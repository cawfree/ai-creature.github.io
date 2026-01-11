import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';

import {Transition} from './@types';
import {ReplyBuffer} from './classes';
import {
  assertNumericArray,
  createAgentSacTrainable,
} from './utils';

void (async () => {
    const DISABLED = false

    const {agent} = await createAgentSacTrainable({
      agentSacProps: {
        batchSize: 100,
        verbose: true,
      },
    });

    const actor = agent.actor;
    assert(actor);

    actor.summary();

    // eslint-disable-next-line no-restricted-globals
    self.postMessage({weights: await Promise.all(actor.getWeights().map(w => w.array()))});

    const rb = new ReplyBuffer(
      50000,
      ({state: [telemetry, frameL, frameR], action, reward}: Omit<Transition, 'nextState'>) => {
        frameL.dispose();
        frameR.dispose();
        telemetry.dispose();
        action.dispose();
        reward.dispose();
      },
    );

    const job = async () => {
        if (DISABLED) return 99999
        if (rb.size < agent._batchSize*10) return 1000
        
        const samples = rb.sample(agent._batchSize) // time fast
        if (!samples.length) return 1000
    
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
            framesL.push(frameL)
            framesR.push(frameR)
            telemetries.push(telemetry)
            actions.push(action)
            rewards.push(reward)
            nextFramesL.push(nextFrameL)
            nextFramesR.push(nextFrameR)
            nextTelemetries.push(nextTelemetry)
        }
    
       tf.tidy(() => {
          console.time('train')
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
          console.timeEnd('train')
       }); 

        console.time('train postMessage')
        // eslint-disable-next-line no-restricted-globals
        self.postMessage({
          weights: await Promise.all(actor.getWeights().map(w => w.array())),
        });
        console.timeEnd('train postMessage')
    
        return 1
    }
    
    /**
     * Executes job.
     */
    const tick = async () => {
        try {
            setTimeout(tick, await job())
        } catch (e) {
            console.error(e)
            setTimeout(tick, 5000) // show must go on (҂◡_◡) ᕤ
        }
    }
    
    setTimeout(tick, 1000)
    
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
    
        if (i % rb._limit === 0) {
            agent.checkpoint() // timer ~ 500ms, don't await intentionally
        }
    })
})()