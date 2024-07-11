

async function main() {
const model = tf.sequential();

model.add(tf.layers.dense({inputShape: [1], units: 1, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'softmax'}));

model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError',
    metrics: ['accuracy']
});


const data = tf.tensor1d([0, 5, 10, 20]);
const labels = tf.tensor1d([32, 37, 42, 68]);

await model.fit(data, labels, {
    epochs: 100,
    callbacks: {
        onEpochEnd: async (epoch, logs) => {
            console.log('onEpochEnd  ' + epoch + JSON.stringify(logs));
        }
    }
})

// console.log(JSON.stringify(model.outputs[0]))

model.predict(tf.tensor1d([15])).print()

}


main()


















































// async function main() {
//     const model = await tf.sequential();

//     model.add(tf.layers.dense({inputShape: [784], units: 10, activation: 'relu'}));
//     model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

//     model.compile({
//         optimizer: 'sgd',
//         loss: 'categoricalCrossentropy',
//         metrics: ['accuracy']
//     });


//     const data = tf.tensor([10, 784]);
//     const labels = tf.randomUniform([10, 10]);


//     await model.fit(data, labels, {
//         epochs: 100 ,
//         batchSize: 32,
//         callbacks: {
//             onTrainBegin: async () => {
//                 console.log('onTrainBegin')
//             },
//             onTrainEnd: async () => {
//                 console.log('onTrainEnd')
//             },
//             onEpochBegin: async (epoch, logs) => {
//                 console.log('onEpochBegin ' + epoch + JSON.stringify(logs));
//             },
//             onEpochEnd: async (epoch, logs) => {
//                 console.log('onEpochEnd ' + epoch + JSON.stringify(logs))
//             },
//             onBatchBegin: async (epoch, logs) => {
//                 console.log('onBatchBegin ' + epoch, JSON.stringify(logs))
//             }, 
//             onBatchEnd: async (epoch, logs) => {
//                 console.log('onBatchEnd ' + epoch + JSON.stringify(logs))
//             }
//         }
//     });
//     model.predict(tf.randomNormal( 3, 784)).print()
// }

// main();