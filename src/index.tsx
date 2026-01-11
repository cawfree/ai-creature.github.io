import React from 'react';
import ReactDOM from 'react-dom/client';
import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import './index.css';
import reportWebVitals from './reportWebVitals';

import {Transition} from './@types';
import {AgentSac} from './classes'
import {creature} from './examples';
import { createAgentSac } from './utils';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(<React.StrictMode />);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

(async () => { 
  const {agent} = await createAgentSac({
    agentSacProps: {
      trainable: false,
      verbose: false,
    },
  });

  const worker = new Worker(new URL('./worker.ts', import.meta.url), {type: 'module'});

  let inited = false;
  let busy = false;

  void worker.addEventListener('message', e => {
    const {weights/*, frame */} = e.data;
    if (!weights) return;

    assert(Array.isArray(weights));

    void tf.tidy(() => {
      inited = true;
      return void agent.actor!.setWeights(weights.map(w => tf.tensor(w)));
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
    agent,
    onTransitionPublished: (
      transition: Omit<Transition, 'nextState'>
    ) => worker.postMessage({action: 'newTransition', transition}),
    whileNotBusyWhenReady,
  }); 
})();

