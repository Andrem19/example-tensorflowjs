import * as tf from '@tensorflow/tfjs-node-gpu'


async function prediction() {
// 1. Set Data
const trainData = {
    xs: tf.tensor2d([[2], [5], [13]], [3, 1]),
    ys: tf.tensor2d([[3], [8], [21]], [3, 1])
}

// 2. Create a model
const model = tf.sequential()
model.add(tf.layers.dense({
    units: 1,
    inputShape: [1],
    activation: 'relu'
}))
model.compile({
    loss: "meanSquaredError",
    optimizer: "sgd"
})
model.summary()
// 3. Train model
await model.fit(trainData.xs, trainData.ys, {
    epochs: 500,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch} Loss: ${logs.loss}`)
        }
    }
})
//save data 

await model.save("file:////home/andrew/CODE/JS/MACHINE_LEARNING/tensorflow_1/model")

 }
prediction()