import React from 'react';
import ReactDOM from 'react-dom/client';
import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import './index.css';

import {Transition} from './@types';
import {creature} from './examples';
import {createAgentSacInstance} from './utils';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(<React.StrictMode />);

(async () => { 
  const agentSacInstance = await createAgentSacInstance();
  const worker = new Worker(new URL('./worker.ts', import.meta.url), {type: 'module'});

  let inited = false;
  let busy = false;

  void worker.addEventListener('message', e => {
    const {weights/*, frame */} = e.data;
    if (!weights) return;

    assert(Array.isArray(weights));

    void tf.tidy(() => {
      inited = true;
      return void agentSacInstance.actor.setWeights(weights.map(w => tf.tensor(w)));
    });
  });

  const whileNotBusyWhenReady = (cb: Function) => async (...args: unknown[]) => {
    if (busy || !inited) return;

    try {
      busy = true;
      const res = await cb(...args);
      busy = false;
      return res;
    } catch (e) {
      busy = false;
      throw e;
    }
  };

  return creature.createCreatureEngine({
    agentSacInstance,
    onTransitionPublished: (
      transition: Omit<Transition, 'nextState'>
    ) => worker.postMessage({action: 'newTransition', transition}),
    whileNotBusyWhenReady,
  }); 
})();

