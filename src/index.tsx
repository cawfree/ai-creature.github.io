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

window.engine = null;
window.scene = null;
window.sceneToRender = null;

const agent = new AgentSac({trainable: false, verbose: false})

const canvas = document.getElementById("renderCanvas");
const createDefaultEngine = () => new BABYLON.Engine(canvas, true, {
    preserveDrawingBuffer: true, 
    stencil: true,
    disableWebGL2Support: false
})

window.vote = 0
document.getElementById("like").addEventListener("click", () => {
    // if (!transitions.length) return

    window.reward = 1
    // transitions[transitions.length - 1].reward += reward
    // globalReward += reward
    // console.log('reward like: ', transitions[transitions.length - 1].reward, globalReward)
})

document.getElementById("dislike").addEventListener("click", () => {
    // if (!transitions.length) return

    window.reward = -1
    // transitions[transitions.length - 1].reward += reward
    // globalReward += reward
    // console.log('reward dislike: ', transitions[transitions.length - 1].reward, globalReward)
})

window.transitions = []
window.globalReward = 0
const BINOCULAR = true

const createScene = async () => {
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

    
    // crCameraLeft.rotation = new BABYLON.Vector3(0, -(Math.PI - 0.3), 0)
    // crCameraLeft.fovMode = BABYLON.Camera.PERSPECTIVE_CAMERA;
    // crCameraRight.rotation = new BABYLON.Vector3(0, +(Math.PI - 0.3), 0)
    // crCameraRight.fovMode = BABYLON.Camera.FOVMODE_HORIZONTAL_FIXED;

    // crCameraRight.checkCollisions = true;
    // crCamera.rotation = (new BABYLON.Vector3(0.5, 0, 0))
    // crCamera.ellipsoid = new BABYLON.Vector3(1, 1, 1);
    // crCamera.ellipsoidOffset = new BABYLON.Vector3(3, 3, 3);
    // creature.checkCollisions = true;
    // scene.collisionsEnabled = true;
    // crCamera.applyGravity = true;

    // crCamera.fovMode = BABYLON.Camera.PERSPECTIVE_CAMERA;
    // crCamera.fovMode = BABYLON.Camera.FOVMODE_HORIZONTAL_FIXED;
    // crCamera.inertia = 2
    // crCamera.setTarget(new BABYLON.Vector3(2, 0, 0))
    // const crCameraMesh = BABYLON.MeshBuilder.CreateSphere("cr_camera_mesh", {diameter: 1, segments: 32}, scene);
    // crCameraMesh.parent = crCamera
    // crCameraMesh.isVisible = 1


    /* CLIENT */
    const client = BABYLON.MeshBuilder.CreateSphere("client", {diameter: 3, segments: 32}, scene);
    client.parent = camera
    client.setParent(camera)
    // client.position = new BABYLON.Vector3(0, -12,0)

    const clMat = new BABYLON.StandardMaterial("cl_mat", scene)
    clMat.diffuseColor = new BABYLON.Color3(0, 0, 0)
    client.material = clMat

    engine.hideLoadingUI();

    /* CAGE */
    const cage = BABYLON.MeshBuilder.CreateSphere("cage", {
        segements: 64, 
        diameter: 50
    }, scene)

    // const cage = BABYLON.MeshBuilder.CreateBox("cage", {
    //     width: 100, 
    //     depth: 100, 
    //     height: 40
    // }, scene)
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

    

    /* MIRROR */
    /* const mirror = BABYLON.MeshBuilder.CreateBox("mirror", {
        width: 10, 
        depth: 0.1, 
        height: 5
    }, scene)
    mirror.material = new BABYLON.StandardMaterial("mirror_mat", scene)
    mirror.position = new BABYLON.Vector3(20, 0, 0)
    // mirror.addRotation(0, Math.PI/2, 0)
    mirror.isVisible = true
    // How to use: mirror.material.diffuseTexture = new BABYLON.Texture(base64Data, scene) // timer ~1ms
    */

    // const [ballRed, ballGreen, ballBlue, ballPurple, ballYellow] = ['red', 'green', 'blue', 'purple', 'yellow'].map(color => {
   
    const ballPos = [[-10,-10,10], [10,-10,-10], [-10,-10,-10], [10,-10,10]]
    // const balls = ['red', 'green', 'blue', 'purple'].map((color, i) => {
    const balls = ['green', 'green', 'red', 'red'].map((color, i) => {
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

        return ball
    })

    // balls[0].position = new BABYLON.Vector3(10, 0, 0)

    /* SHuffle */
    // scene.onPointerDown = function(evt, pickInfo) {
    //     if(pickInfo.hit && pickInfo.pickedMesh.id.startsWith('cage')) {
    //         const getRand = () => new BABYLON.Vector3(Math.random()/10 - 0.1, Math.random()/10 - 0.1, Math.random()/10 - 0.1)

    //         balls.forEach(ball => ball.impostor.applyImpulse(getRand(), BABYLON.Vector3.Zero()))
    //     }
    // }

    // setInterval(()=>{
    //     const getRand = () => new BABYLON.Vector3(Math.random()/10 - 0.1, Math.random()/10 - 0.1, Math.random()/10 - 0.1)

    //     balls.forEach(ball => ball.impostor.applyImpulse(getRand(), BABYLON.Vector3.Zero()))
    // }, 1000)


    // ballRed.impostor.applyImpulse(new BABYLON.Vector3(0, -20, 0), BABYLON.Vector3.Zero())
    // ballGr.impostor.applyImpulse(new BABYLON.Vector3(0, -20, 0), BABYLON.Vector3.Zero())


    ///* WORKER */
    let inited = false 
    const worker = new Worker(
      new URL("./worker.ts", import.meta.url),
      { type: "module" }
    )
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
    
    // ;(() => {
    //     let coll
    //     creature.impostor.onCollide = e => {
    //         coll = e.point.subtract(creature.position).normalize()
    //         console.log('crea', coll)
    //         if (window.onCollide)
    //             window.onCollide(coll)
    //     }

    //     balls.forEach(ball => {
    //         ball.impostor.onCollide = e => {
    //             const collision = e.point.subtract(creature.position).normalize()
    //             console.log('crea ball', coll, collision)

    //             if (window.onCollide)
    //                 window.onCollide(collision, ball.id)

    //             // if (ball.id.endsWith('_red'))
    //             console.log('onCollide mesh:', ball.id)
    //         }
    //     })
    // })()



    const base64ToImg = (base64) => new Promise((res, _) => {
        const img = new Image()
        img.src = base64
        img.onload = () => res(img)
    })
    const TRANSITIONS_BUFFER_SIZE = 2
    const frameEvery = 1000/30 // ~33ms ~24frames/sec
    const frameStack = []
    // const transitions = []

    // let start = Date.now() + frameEvery
    let timer = Date.now() 
    let busy = false
    let stateId = 0

    let prevLinearVelocity = BABYLON.Vector3.Zero()
    window.collision = BABYLON.Vector3.Zero()
    window.reward = 0
    window.globalReward = 0
    // let collisionMesh = null

    const testLayer = agent.actor.layers[4]
    const spy = tf.model({inputs: agent.actor.inputs, outputs: testLayer.output})

    scene.registerAfterRender(async () => { // timer ~ 20-90ms
        if (/*Date.now() < start || */busy || !inited) return

        // const delta = (Date.now() - timestamp) / 1000 // sec
        // timestamp = Date.now() 
        // start = Date.now() + frameEvery
        busy = true

        // const timerLbl = 'TimerLabel-' + start
        
        /*
        console.time(timerLbl)
        console.timeEnd(timerLbl)
        console.log('numTensors BEFORE: ' + tf.memory().numTensors)
        console.log('numTensors AFTER: ' + tf.memory().numTensors)
        */







        // const screenShots = []
        // screenShots.push(
        //     BABYLON.Tools.CreateScreenshotUsingRenderTargetAsync(engine, crCameraLeft, { // ~ 7-60ms
        //         height: agent._frameShape[0],
        //         width: agent._frameShape[1]
        //     })
        // )
        // screenShots.push(
        //     BABYLON.Tools.CreateScreenshotUsingRenderTargetAsync(engine, crCameraRight, { // ~ 7-60ms
        //         height: agent._frameShape[0],
        //         width: agent._frameShape[1]
        //     })
        // )
        // const base64Data = await Promise.all(screenShots)
        // frameStack.push(base64Data)




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
                const greyScaler = tf.tensor([0.299, 0.587, 0.114], [1, 1, 3])
                let imgTensors = imgs.map(img => tf.browser.fromPixels(img)
                    //.mul(greyScaler).sum(-1, true)
                )

                // optic chiasma
                // imgTensors = imgTensors.map(img => tf.split(img, 2, 1))
                // for (let i = 0; i < imgTensors.length; i = i + 2) {
                //     const first = tf.concat([imgTensors[i][0], imgTensors[i+1][0]], -1)
                //     const second = tf.concat([imgTensors[i][1], imgTensors[i+1][1]], -1)
                //     imgTensors[i] = first
                //     imgTensors[i+1] = second
                // }
                
                // imgTensors = [
                //     imgTensors[0].concat(imgTensors[1], 1), 
                //     //imgTensors[2].concat(imgTensors[3], 1)
                // ]


                // if (collisionMesh) {
                    imgTensors = imgTensors.map((t, i) => {
                        const canv = document.getElementById('testCanvas' + (i+3))
                        if (canv) {
                            tf.browser.toPixels(t, canv) // timer ~1ms
                        }
                        return t
                            .sub(255/2)
                            .div(255/2) 
                    })
                // }

                const resL = tf.concat(imgTensors.filter((el, i) => i%2==0), -1)
                const resR = tf.concat(imgTensors.filter((el, i) => i%2==1), -1)
                return [resL, resR]

                // return [tf.concat(imgTensors, -1)]

                // let frTest = tf.unstack(res, -1)
                //     // frTest = [tf.concat(frTest.slice(0,3), -1), tf.concat(frTest.slice(3), -1)]
                // console.log(frTest[0].arraySync()[30][0][0], frTest[3].arraySync()[30][0][0])
                
                // console.log(tf.concat(tf.unstack(tf.concat(imgTensors, 2), -1), -1).arraySync()[30][0][0])

            })
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
            ]
            const reward = window.reward

            //collisionMesh &&  
            // if (collisionMesh && transitions.length) {
            //     tf.tidy(() => {
            //         let frTest = tf.unstack(tf.tensor(transitions[transitions.length - 1].state[1], [64,128, agent._nFrames]), -1)
            //         // frTest = [tf.stack(frTest.slice(0,3), -1), tf.stack(frTest.slice(3), -1)]
            //         let i = 0
            //         for (const fr of frTest) {
            //             i++
            //             tf.browser.toPixels(fr, document.getElementById('testCanvas' + i)) // timer ~1ms
            //         }
            //     })
            // }

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

                let tiles = tf.clipByValue(tf.squeeze(testOutput), 0, 1)
                tiles = tf.transpose(tiles, [2,0,1])
                tiles = tf.unstack(tiles)

                let res = [], line = []
                for (const [i, tile] of tiles.entries()) {
                    line.push(tile)
                    if ((i+1) % 8 == 0 && i) {
                        res.push(tf.concat(line, 1))
                        line = []
                    }
                }
                const testFr = tf.concat(res)
                tf.browser.toPixels(testFr, document.getElementById('testCanvas2')) // timer ~1ms
            })

            const 
                impulse = actionArr.slice(0, 3)//.map(el => el/10)//, // [0,-1, 0], //    
                // rotation = actionArr.slice(3, 7).map(el => el),
                // color = actionArr.slice(3, 6).map(el => el)/.map(el => el) // [-1,1] => [0,2] => [0, 255]
                // look = actionArr.slice(3, 6)

            // console.log('tel tel: ', telemetry.map(t=> t.toFixed(3)))
            // console.log('tel imp:', impulse.map(t=> t.toFixed(3)))

            console.assert(actionArr.length === 3, actionArr.length)
            console.assert(impulse.length === 3)
            // console.assert(look.length === 3)
            // console.assert(rotation.length === 4)
            // console.assert(color.length === 3)

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
            }
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

                // if (transition.reward > 0) {
                //     transition.priority = 7
                //     console.log('reward prio:', transition, transition.state[0])
                // }
                window.globalReward += transition.reward
                console.log('reward', transition.reward, window.globalReward)
                

                worker.postMessage({action: 'newTransition', transition}) // timer ~ 6ms

            }

            // imgTensors.forEach(t => t.dispose())
            // frames.dispose()
            framesNorm.map(fr => fr.dispose())
            framesBatch.map(fr => fr.dispose())
            telemetryBatch.dispose()
            action.dispose()

            // if (stateId%1 == 0)
            //     frameStack.forEach((base64Data, i) => {
            //         const img = new Image()
            //         img.onload = () => document.getElementById('testCanvas' + (i+2))
            //             .getContext('2d')
            //             .drawImage(img, 0, 0, 256, 128)
            //         img.src = base64Data
            //     })

            frameStack.length = 0 // I will regret about this :D
        }

        //mirror.material.diffuseTexture = new BABYLON.Texture(base64Data, scene) // timer ~1ms
        
        // const img = await base64ToImg(base64Data) // timer ~2-12ms
        // const tensor = tf.browser.fromPixels(img) // timer ~ 1ms
        // const arr = await tensor.array() // timer ~ 6-15ms
        // worker.postMessage(arr) // timer ~ 6ms
        // tensor.dispose()
        
        busy = false
    })

    return scene
};

window.initFunction = async () => {
  await Ammo();
  
  window.engine = createDefaultEngine();
  assert(engine);

  window.scene = await createScene();
};

void initFunction().then(() => {
  sceneToRender = scene;
  engine.runRenderLoop(() => {
    if (!sceneToRender?.activeCamera) return;
    return sceneToRender.render();
  });
});

window.addEventListener('resize', () => void engine.resize());
