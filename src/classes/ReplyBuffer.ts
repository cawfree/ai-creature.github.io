import {Transition} from '../@types';
import {getRandomInt} from '../utils';

export class ReplyBuffer {

  _limit: number;
  _onDiscard: (transition: Transition) => void;
  size: number;
  _pool: number[];
  _buffer: Transition[];

  /**
   * Constructor.
   * 
   * @param {*} limit maximum number of transitions
   * @param {*} onDiscard callback triggered on discard a transition
   */
  constructor(limit = 500, onDiscard = () => {}) {
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
  add(transition: Transition) {
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