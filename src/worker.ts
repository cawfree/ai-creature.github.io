import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';

import {Transition} from './@types';
import {assertNumericArray} from './utils';

import {CreatureState} from './examples/creature/@types';
import {createCreatureAgentSacTrainableInstance} from './examples/creature/utils';

// https://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
const getRandomInt = (min: number, max: number)  => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

const DISABLED = false
const BATCH_SIZE_AMPLIFIER = 10;

class ReplyBuffer<State> {

  _limit: number;
  _onDiscard: (transition: Omit<Transition<State>, 'nextState'>) => void;
  size: number;
  _pool: number[];
  _buffer: Omit<Transition<State>, 'nextState'>[];

  /**
   * Constructor.
   * 
   * @param {*} limit maximum number of transitions
   * @param {*} onDiscard callback triggered on discard a transition
   */
  constructor(limit = 500, onDiscard = (transition: Omit<Transition<State>, 'nextState'>) => {}) {
      this._limit = limit
      this._onDiscard = onDiscard

      this._buffer = new Array(limit).fill(null)
      this._pool = []

      this.size = 0
  }

  /**
   * Add a new transition to the reply buffer. 
   * Transition doesn't contain the next state. The next state is derived when sampling.
   * 
   * @param {{id: number, priority: number, state: array, action, reward: number}} transition transition
   */
  add(transition: Omit<Transition<State>, 'nextState'>) {
      let { id, priority = 1 } = transition
      if (id === undefined || id < 0 || priority < 1) 
          throw new Error('Invalid arguments')

      id = id % this._limit

      if (this._buffer[id]) {
          const start = this._pool.indexOf(id)
          let deleteCount = 0
          while (this._pool[start + deleteCount] == id)
              deleteCount++

          this._pool.splice(start, deleteCount)
          
          this._onDiscard(this._buffer[id])
      } else
          this.size++

      while (priority--) 
          this._pool.push(id)

      this._buffer[id] = transition
  }

  /**
   * Return `n` random samples from the buffer. 
   * Returns an empty array if impossible provide required number of samples.
   * 
   * @param {number} [n = 1] - number of samples 
   * @returns array of exactly `n` samples
   */
  sample(n: number = 1) {
      if (this.size < n) 
          return []

      const 
          sampleIndices = new Set(),
          samples = []

      let counter = n
      while (counter--)
          while (sampleIndices.size < this.size) {
              const randomIndex = this._pool[getRandomInt(0, this._pool.length - 1)]
              if (sampleIndices.has(randomIndex))
                  continue

              sampleIndices.add(randomIndex)

              const { id, state, action, reward } = this._buffer[randomIndex]
              const nextId = id + 1
              const next = this._buffer[nextId % this._limit]

              if (next && next.id === nextId) { // the case when sampled the last element that still waiting for next state
                  samples.push({ state, action, reward, nextState: next.state})
                  break
              }
          }

      return samples.length == n ? samples : []
  }
}

void (async () => {

  const agentSacTrainableInstance =
    await createCreatureAgentSacTrainableInstance({
      agentSacProps: {
        batchSize: 100,
      },
    });

  // eslint-disable-next-line no-restricted-globals
  self.postMessage({
    weights: await Promise.all(agentSacTrainableInstance.actor.getWeights().map(w => w.array())),
  });

  const rb = new ReplyBuffer<CreatureState>(
     5000 * agentSacTrainableInstance.batchSize,
    ({state: [telemetry, frameL, frameR], action, reward}: Omit<Transition<CreatureState>, 'nextState'>) => {
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
      void agentSacTrainableInstance.train({transitions: samples});
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
    const decodeTransition = (transition: Record<string, unknown>): Omit<Transition<CreatureState>, 'nextState'> => {
        const {id, state, action, reward, priority} = transition

        assert(typeof id === 'number');
        assert(typeof reward === 'number');

        assert(priority === undefined || typeof priority === 'number');

        assert(Array.isArray(state));
        assert(Array.isArray(action)); // [number, number, number]

        const [telemetry, frameL, frameR] = state;
    
        return tf.tidy((): Omit<Transition<CreatureState>, 'nextState'> => ({
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