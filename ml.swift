import TensorFlow

// Define the input data
let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
let y: Tensor<Float> = [[0], [1], [1], [0]]

// Define the model
struct XOR: Layer {
    var layer1 = Dense<Float>(inputSize: 2, outputSize: 8, activation: relu)
    var layer2 = Dense<Float>(inputSize: 8, outputSize: 1)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let h1 = layer1(input)
        return layer2(h1)
    }
}

// Define the optimizer
var model = XOR()
let optimizer = SGD(for: model, learningRate: 0.1)

// Train the model
for epoch in 1...10000 {
    let ŷ = model(x)
    let loss = meanSquaredError(predicted: ŷ, expected: y)
    if epoch % 1000 == 0 {
        print("Epoch: \(epoch) Loss: \(loss)")
    }
    let 𝛁loss = gradient(at: model) { model -> Tensor<Float> in
        let ŷ = model(x)
        return meanSquaredError(predicted: ŷ, expected: y)
    }
    optimizer.update(&model, along: 𝛁loss)
}

// Test the model
let testX: Tensor<Float> = [[0, 1], [1, 0], [0, 0], [1, 1]]
let testY = model(testX)
print(testY)
