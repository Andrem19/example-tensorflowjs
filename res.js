import * as tf from '@tensorflow/tfjs-node-gpu'

const predict = async () => {
const loadingModel = await tf.loadLayersModel('file:////home/andrew/CODE/JS/MACHINE_LEARNING/tensorflow_1/model/model.json');

// 4. Make a prediction
const testPredictValue = tf.tensor2d([4181], [1, 1])
const prediction = await loadingModel.predict(testPredictValue).data()

console.log(prediction);
}
predict()