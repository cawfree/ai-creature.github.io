// @ts-nocheck

import React from 'react';
import ReactDOM from 'react-dom/client';
import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import './index.css';
import reportWebVitals from './reportWebVitals';

import {AgentSac} from './classes'

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(<React.StrictMode />);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

const agent = new AgentSac({trainable: false, verbose: false})

const canvas = document.getElementById("renderCanvas");
const createDefaultEngine = () => new BABYLON.Engine(canvas, true, {
    preserveDrawingBuffer: true, 
    stencil: true,
    disableWebGL2Support: false
})

document.getElementById('like').addEventListener('click', () => (
  window.reward = 1
));

document.getElementById('dislike').addEventListener('click', () => {
  window.reward = -1
});

window.transitions = []
const BINOCULAR = true

const createScene = async ({
  engine,
}: {
  readonly engine: BABYLON.Engine;
}) => {
    await agent.init()

    // This creates a basic Babylon Scene object (non-mesh)
    const scene = new BABYLON.Scene(engine);
    scene.collisionsEnabled = true

    // Environment
    const hdrTexture = BABYLON.CubeTexture.CreateFromPrefilteredData("https://ai-creature.github.io/3d/env/environment.dds", scene);
    hdrTexture.name = "envTex";
    hdrTexture.gammaSpace = false;
    scene.environmentTexture = hdrTexture;

    const skybox = BABYLON.MeshBuilder.CreateBox("skyBox", {size:1000.0}, scene);
    const skyboxMaterial = new BABYLON.StandardMaterial("skyBox", scene);
    skyboxMaterial.backFaceCulling = false;
    skyboxMaterial.reflectionTexture = new BABYLON.CubeTexture("https://ai-creature.github.io/3d/env/skybox", scene);
    skyboxMaterial.reflectionTexture.coordinatesMode = BABYLON.Texture.SKYBOX_MODE;
    skyboxMaterial.diffuseColor = new BABYLON.Color3(0, 0, 0);
    skyboxMaterial.specularColor = new BABYLON.Color3(0, 0, 0);
    skybox.material = skyboxMaterial;

    //CAMERA
    const camera = new BABYLON.ArcRotateCamera("Camera", BABYLON.Tools.ToRadians(-120), BABYLON.Tools.ToRadians(80), 65, new BABYLON.Vector3(0, -15, 0), scene);    
    camera.attachControl(canvas, true);
    camera.lowerRadiusLimit = 10;
    camera.upperRadiusLimit = 120;
    camera.collisionRadius = new BABYLON.Vector3(2, 2, 2);
    camera.checkCollisions = true;

    //enable Physics in the scene vector = gravity
    scene.enablePhysics(new BABYLON.Vector3(0, 0, 0), new BABYLON.AmmoJSPlugin(false));

    const physicsEngine = scene.getPhysicsEngine()
    // physicsEngine.setSubTimeStep(physicsEngine.getTimeStep()/3 * 1000)
    physicsEngine.setTimeStep(1 / 60)
    physicsEngine.setSubTimeStep(1)

    //LIGHTS
    const light1 = new BABYLON.PointLight("light1", new BABYLON.Vector3(0, 5,-6), scene);
    const light2 = new BABYLON.PointLight("light2", new BABYLON.Vector3(6, 5, 3.5), scene);
    const light3 = new BABYLON.DirectionalLight("light3", new BABYLON.Vector3(20, -5, 20), scene);
    light1.intensity = 15;
    light2.intensity = 5;

    engine.displayLoadingUI();

    await Promise.all([
        BABYLON.SceneLoader.AppendAsync("https://ai-creature.github.io/3d/marbleTower.glb"),
        BABYLON.SceneLoader.AppendAsync("https://models.babylonjs.com/Marble/marble/marble.gltf")
    ])
    scene.getMeshByName("marble").isVisible = false

    const tower = scene.getMeshByName("tower");
    tower.setParent(null)
    tower.checkCollisions = true;
    tower.impostor = new BABYLON.PhysicsImpostor(tower, BABYLON.PhysicsImpostor.MeshImpostor, {
        mass: 0,
        friction: 1
    }, scene);
    tower.material = scene.getMaterialByName("stone")
    tower.material.backFaceCulling = false
    

    /* CREATURE */
    const creature = BABYLON.MeshBuilder.CreateSphere("creature", {diameter: 1, segments:32}, scene)
    creature.parent = null
    creature.setParent(null)
    creature.position = new BABYLON.Vector3(0,-5,0)

    creature.isPickable = false

    const crMat = new BABYLON.StandardMaterial("cr_mat", scene);
    crMat.alpha = 0 // for screenshots
    creature.material = crMat

    creature.impostor = new BABYLON.PhysicsImpostor(creature, BABYLON.PhysicsImpostor.SphereImpostor, {
        mass: 1,
        friction: 0,
        stiffness: 0,
        restitution: 0
    }, scene)
    
    BABYLON.ParticleHelper.SnippetUrl = "https://ai-creature.github.io/3d/snippet";
    // Sparks
    creature.sparks = await BABYLON.ParticleHelper.CreateFromSnippetAsync("UY098C-3.json", scene, false);
    creature.sparks.emitter = creature;
    // Core
    creature.glow = await BABYLON.ParticleHelper.CreateFromSnippetAsync("EXUQ7M-5.json", scene, false);
    creature.glow.emitter = creature;

    /* CREATURE's CAMERA */
    const crCameraLeft = new BABYLON.UniversalCamera("cr_camera_l", new BABYLON.Vector3(0, 0, 0), scene)
    crCameraLeft.parent = creature
    crCameraLeft.position = new BABYLON.Vector3(-0.5, 0, 0)//new BABYLON.Vector3(0, 5, -10)
    crCameraLeft.fov = 2
    crCameraLeft.setTarget(new BABYLON.Vector3(-1, 0, 0.6))

    const crCameraRight = new BABYLON.UniversalCamera("cr_camera_r", new BABYLON.Vector3(0, 0, 0), scene)
    crCameraRight.parent = creature
    crCameraRight.position = new BABYLON.Vector3(0.5, 0, 0)//new BABYLON.Vector3(0, 5, -10)
    crCameraRight.fov = 2
    crCameraRight.setTarget(new BABYLON.Vector3(1, 0, 0.6))



    const crCameraLeftPl = BABYLON.MeshBuilder.CreateSphere("crCameraLeftPl", {diameter: 0.1, segments: 32}, scene);
    crCameraLeftPl.parent = creature
    crCameraLeftPl.position = new BABYLON.Vector3(-0.5, 0, 0)
    const crCameraLeftPlclMat = new BABYLON.StandardMaterial("crCameraLeftPlclMat", scene)
    crCameraLeftPlclMat.alpha = 0.3 // for screenshots
    crCameraLeftPlclMat.diffuseColor = new BABYLON.Color3(0, 0, 0)
    crCameraLeftPl.material = crCameraLeftPlclMat

    const crCameraRightPl = BABYLON.MeshBuilder.CreateSphere("crCameraRightPl", {diameter: 0.1, segments: 32}, scene);
    crCameraRightPl.parent = creature
    crCameraRightPl.position = new BABYLON.Vector3(0.5, 0, 0)
    const crCameraRightPlclMat = new BABYLON.StandardMaterial("crCameraRightPlclMat", scene)
    crCameraRightPlclMat.alpha = 0.3 // for screenshots
    crCameraRightPlclMat.diffuseColor = new BABYLON.Color3(0, 0, 0)
    crCameraRightPl.material = crCameraRightPlclMat

    /* CLIENT */
    const client = BABYLON.MeshBuilder.CreateSphere("client", {diameter: 3, segments: 32}, scene);
    client.parent = camera
    client.setParent(camera)

    const clMat = new BABYLON.StandardMaterial("cl_mat", scene)
    clMat.diffuseColor = new BABYLON.Color3(0, 0, 0)
    client.material = clMat

    engine.hideLoadingUI();

    /* CAGE */
    const cage = BABYLON.MeshBuilder.CreateSphere("cage", {
        segements: 64, 
        diameter: 50
    }, scene)

    cage.parent = null
    cage.setParent(null)
    cage.position = new BABYLON.Vector3(0, -12,0)
    cage.isPickable = true

    const cageMat = new BABYLON.StandardMaterial("cage_mat", scene);
    cageMat.alpha = 0.1 // for ray hit
    cage.material = cageMat
    cage.material.backFaceCulling = false

    cage.impostor = new BABYLON.PhysicsImpostor(cage, BABYLON.PhysicsImpostor.MeshImpostor, {
        mass: 0,
        friction: 1
    }, scene);

    const ballPos = [[-10,-10,10], [10,-10,-10], [-10,-10,-10], [10,-10,10]]
    void ['green', 'green', 'red', 'red'].forEach((color, i) => {
        const ball = BABYLON.MeshBuilder.CreateSphere("ball_"+ color + i, {diameter: 7, segments: 64}, scene)
        ball.position = new BABYLON.Vector3(...ballPos[i])
        ball.parent = null
        ball.setParent(null)
        ball.isPickable = true
        ball.impostor = new BABYLON.PhysicsImpostor(ball, BABYLON.PhysicsImpostor.SphereImpostor, {
            mass: 7,
            friction: 1,
            stiffness: 1,
            restitution: 1
        }, scene);
        ball.material = scene.getMaterialByName(color + "Mat")
        ball.checkCollisions = true
        ball.material.backFaceCulling = false
    });

    /* WORKER */
    let inited = false 
    const worker = new Worker(new URL('./worker.ts', import.meta.url), {type: 'module'});

    worker.addEventListener('message', e => {
        const { weights, frame } = e.data

        tf.tidy(() => {
            if (weights) {
                inited = true
                agent.actor.setWeights(weights.map(w => tf.tensor(w))) // timer ~30ms
                if (Math.random() > 0.99) console.log('weights:', weights)
            }

        })
    })

    /* COLLISIONS DETECTION */
    const impostors = scene.getPhysicsEngine()._impostors.filter(im => im.object.id !== creature.id)
    creature.impostor.registerOnPhysicsCollide(impostors, (body1, body2) => {})
    impostors.forEach(impostor => {
        impostor.onCollide = e => {
            if (window.onCollide) {
                const collision = e.point.subtract(creature.position).normalize()
                window.onCollide(collision, impostor.object.id)
            }
        }
    })

    const base64ToImg = (base64) => new Promise((res, _) => {
        const img = new Image()
        img.src = base64
        img.onload = () => res(img)
    })
    const TRANSITIONS_BUFFER_SIZE = 2
    const frameEvery = 1000/30 // ~33ms ~24frames/sec
    const frameStack = []

    let timer = Date.now() 
    let busy = false
    let stateId = 0

    let prevLinearVelocity = BABYLON.Vector3.Zero()
    window.collision = BABYLON.Vector3.Zero()
    window.reward = 0

    const testLayer = agent.actor.layers[4]
    const spy = tf.model({inputs: agent.actor.inputs, outputs: testLayer.output})

    scene.registerAfterRender(async () => { // timer ~ 20-90ms
        if (busy || !inited) return
        busy = true

        //delay
        if (!frameStack.length) {
            frameStack.push([
                await BABYLON.Tools.CreateScreenshotUsingRenderTargetAsync(engine, crCameraLeft, { // ~ 7-60ms
                    height: agent._frameShape[0],
                    width: agent._frameShape[1]
                })
            ])
        } else {
            frameStack[0].push(
                await BABYLON.Tools.CreateScreenshotUsingRenderTargetAsync(engine, crCameraRight, { // ~ 7-60ms
                    height: agent._frameShape[0],
                    width: agent._frameShape[1]
                })
            )
        }

        if (frameStack.length >= agent._nFrames && frameStack[0].length == 2) { // ~20ms
            if (frameStack.length > agent._nFrames)
                throw new Error("(⊙＿⊙')")

            const imgs = await Promise.all(frameStack.flat().map(fr => base64ToImg(fr)))

            const framesNorm = tf.tidy(() => {
                const greyScaler = tf.tensor([0.299, 0.587, 0.114], [1, 1, 3]);
                const imgTensors = imgs
                    .map(img => tf.browser.fromPixels(img))
                    .map((t, i) => {
                        const canv = document.getElementById('testCanvas' + (i+3))
                        if (canv) tf.browser.toPixels(t, canv); // timer ~1ms
                        return t.sub(255/2).div(255/2);
                    });

                const resL = tf.concat(imgTensors.filter((el, i) => i%2==0), -1)
                const resR = tf.concat(imgTensors.filter((el, i) => i%2==1), -1)
                return [resL, resR]
            });

            const framesBatch = framesNorm.map(fr => tf.stack([fr]))

            const delta = (Date.now() - timer) / 1000 // sec
            console.log('delta (s)', delta)
            const linearVelocity = creature.impostor.getLinearVelocity()
            const linearVelocityNorm = linearVelocity.normalize()
            const acceleration = linearVelocity.subtract(prevLinearVelocity).scale(1/delta).normalize()

            timer = Date.now()
            prevLinearVelocity = linearVelocity

            const ray = new BABYLON.Ray(creature.position, linearVelocityNorm)
            const hit = scene.pickWithRay(ray)
            let lidar = 0
            if (hit.pickedMesh) {
                lidar = Math.tanh((hit.distance - creature.impostor.getRadius())/10) // stretch tanh by 10 for precision
                // console.log('Hit: ', hit.pickedMesh.name, hit.distance, lidar, linearVelocity, collision)
            }

            const telemetry = [
                linearVelocityNorm.x,
                linearVelocityNorm.y,
                linearVelocityNorm.z,
                acceleration.x,
                acceleration.y,
                acceleration.z,
                window.collision.x, 
                window.collision.y, 
                window.collision.z,
                lidar
            ];
            const reward = window.reward

            window.collision = BABYLON.Vector3.Zero() // reset collision point
            window.reward = -0.01
            window.onCollide = undefined
            const telemetryBatch = tf.tensor(telemetry, [1, agent._nTelemetry])
            const action = agent.sampleAction([telemetryBatch, ...framesBatch]) // timer ~5ms

            // TODO: !!!!!await find the way to avoid framesNorm.array()
            console.time('await')
            const [framesArrL, framesArrR,[actionArr]] = await Promise.all([...(framesNorm.map(fr => fr.array())), action.array()]) // action come as a batch of size 1
            console.timeEnd('await')
            // DEBUG Conv encoder
            tf.tidy(() => { // timer ~2.5ms
                const testOutput = spy.predict([telemetryBatch, ...framesBatch], {batchSize: 1})
                console.log('spy', testLayer.name, testOutput.arraySync())
                return
            });

            const impulse = actionArr.slice(0, 3);
            console.assert(actionArr.length === 3, actionArr.length)
            console.assert(impulse.length === 3)

            // [0,-1,0]
            creature.impostor.setAngularVelocity(BABYLON.Quaternion.Zero()) // just in case, probably redundant
            // creature.impostor.setLinearVelocity(BABYLON.Vector3.Zero()) // contact point zero
            creature.impostor.applyImpulse(new BABYLON.Vector3(...impulse), creature.getAbsolutePosition()) // contact point zero
            creature.impostor.setAngularVelocity(BABYLON.Quaternion.Zero())
            // creature.glow.color2 = new BABYLON.Color4(...color)
            
            // after applyImpulse the linear velocity is recalculated right away
            const newLinearVelocity = creature.impostor.getLinearVelocity().normalize() 
            // creature.lookAt(new BABYLON.Vector3(0, -1, 0), 0, 0, 0, BABYLON.Space.LOCAL)
            creature.lookAt(creature.position.add(newLinearVelocity))
            //if (!window.rr) window.rr = 
            // creature.lookAt(creature.position.add(new BABYLON.Vector3(0,1,0)))

            const transtion = {
                id: stateId++, 
                state: [telemetry, framesArrL, framesArrR], // 20ms vs 50ms || size 200kb vs 1.5mb
                action: actionArr,
                reward
            };
            transitions.push(transtion)

            window.onCollide = (collision, mesh) => {
                window.collision = collision
                window.reward += -0.05

                if (mesh.startsWith('ball_')) {
                    console.log('reward', mesh)
                    window.reward = 1

                    if (mesh.includes('red'))
                        window.reward = -1
                }

                window.onCollide = undefined
            }

            if (transitions.length >= TRANSITIONS_BUFFER_SIZE) {
                if (transitions.length > TRANSITIONS_BUFFER_SIZE || TRANSITIONS_BUFFER_SIZE < 2)
                    throw new Error("(⊙＿⊙')")

                const transition = transitions.shift()
                console.log('reward', transition.reward);
                

                worker.postMessage({action: 'newTransition', transition}) // timer ~ 6ms

            }

            framesNorm.map(fr => fr.dispose())
            framesBatch.map(fr => fr.dispose())
            telemetryBatch.dispose()
            action.dispose()

            frameStack.length = 0 // I will regret about this :D
        }

        busy = false
    })

    return scene
};

(async () => {
  await Ammo();
  
  const engine = createDefaultEngine();
  window.addEventListener('resize', () => void engine.resize());

  const sceneToRender = await createScene({engine});
  
  return engine.runRenderLoop(() => {
    if (!sceneToRender?.activeCamera) return;
    return sceneToRender.render();
  });
})();

