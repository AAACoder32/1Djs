/**
 * This method iterate over all nested to check its correct shape
 * @param {data} data 
 * @returns true if data is in correct shape
 */
function checkIsErrorInDataShape(data) {
    let isError = false;
    function getRecursively(data) {
        if (!Array.isArray(data)) {
            return;
        }

        for (let i = 1; i < data.length; i++) {
            if (Array.isArray(data[i - 1]) && Array.isArray(data[i])) {
                if (data[i - 1].length !== data[i].length) {
                    console.error(`${data[i - 1].length} not equals to ${data[i].length}`);
                    isError = true;
                }
            }
        }

        for (let i = 0; i < data.length; i++) {
            getRecursively(data[i]);
        }
    }

    getRecursively(data);
    return isError;
}

/**
 * This method broadcast shape if it is broadcastable
 * @param {Array} shape1 
 * @param {Array} shape2 
 * @returns broadcasted shape
 */
function broadcastShapes(shape1, shape2) {
    // Input validation
    if (!Array.isArray(shape1) || !Array.isArray(shape2)) {
        throw new Error("Inputs must be arrays");
    }
    if (!shape1.every((size) => Number.isInteger(size) && size > 0) ||
        !shape2.every((size) => Number.isInteger(size) && size > 0)) {
        throw new Error("Array elements must be positive integers");
    }

    //Edge cases
    if (shape1.length === 0 || shape2.length === 0) {
        throw new Error("Arrays cannot be empty");
    }
    if (shape1.length === 1 && shape2.length === 1) {
        return [Math.max(shape1[0], shape2[0])];
    }

    const maxLength = Math.max(shape1.length, shape2.length);
    const resultShape = [];

    for (let i = 0; i < maxLength; i++) {
        const size1 = shape1[shape1.length - 1 - i] || 1;
        const size2 = shape2[shape2.length - 1 - i] || 1;

        if (size1 !== size2 && size1 !== 1 && size2 !== 1) {
            throw new Error(`Incompatible shapes at index ${i}`);
        }

        // Only append the maximum size if both sizes are greater than 1
        if (size1 > 1 || size2 > 1) {
            resultShape.push(Math.max(size1, size2));
        }
    }

    return resultShape.reverse();
}

/**
 * Validate target shape broadcast target shape
 * @param {Array} oriShape 
 * @param {Array} targetShape 
 */

function validateBroadcastTargetShape(oriShape, targetShape) {
    if (oriShape.length > targetShape.length) {
        throw new Error("ndim of target shape is less than original shape ndim")
    }

    if (oriShape.length < targetShape.length) {
        const extraDims = targetShape.length - oriShape.length
        const subTargetShape = targetShape.slice(extraDims)
        oriShape.forEach((v, i)=>{
            if(v !== subTargetShape[i] && v !==1){
                throw new Error("Invalid target shape")
            }
        })
    }else{
        oriShape.forEach((v, i)=>{
            if(v !== targetShape[i] && v !==1){
                throw new Error("Invalid target shape")
            }
        })
    }
}


/**
 * This methods takes 1d array and fills the array with its element according to its shape
 * @param {arr} arr 
 * @param {shape} shape 
 * @returns 1d array
 */
function fillArray(arr, shape) {
    let result = [];
    let flatIndex = 0;

    function recursiveFill(shape, index) {
        if (shape.length === 0) {
            result.push(arr[flatIndex % arr.length]);
            flatIndex++;
        } else {
            for (let i = 0; i < shape[0]; i++) {
                recursiveFill(shape.slice(1), index * shape[0] + i);
            }
        }
    }

    recursiveFill(shape, 0);
    return result;
}


/**
 * Takes 1d array with its shape and prints the in readable format
 * @param {arr} arr 
 * @param {shape} shape 
 * @param {indent} indent 
 */
function printArray(arr, shape, indent = 0) {
    if (shape.length === 1) {
        console.log(' '.repeat(indent) + '[' + arr.join(' ') + ']');
    } else {
        console.log(' '.repeat(indent) + '[');
        for (let i = 0; i < shape[0]; i++) {
            printArray(arr.slice(i * shape.slice(1).reduce((a, b) => a * b, 1), (i + 1) * shape.slice(1).reduce((a, b) => a * b, 1)), shape.slice(1), indent + 2);
        }
        console.log(' '.repeat(indent) + ']');
    }
}

/**
 * Takes 1d array of its nd shape
 * @param {shape} shape 
 * @param {arr} arr 
 * @returns multi dimensional array
 */
function makeMultiDimentional(shape, arr) {
    let flatIndex = 0;
    function recursiveFill(shape) {
        if (shape.length === 1) {
            return Array(shape[0]).fill(0).map(() => arr[flatIndex++ % arr.length]);
        } else {
            return Array(shape[0]).fill(0).map(() => recursiveFill(shape.slice(1)));
        }
    }
    return recursiveFill(shape);
}

/**
 * Takes args of 1d arrays
 * @param  {...any} arrays should be 1d arrays
 * @returns as name suggest
 */
function cartesianProduct(...arrays) {
    return arrays.reduce((acc, array) => {
        return acc.flatMap((prev) =>
            array.map((curr) => [...prev, curr])
        )
    }, [[]])
}

/**
 * Takes positive integer
 * @param {x} x 
 * @returns from 0 to n-1 values 1d array
 */
function createIndices(x) {
    let indices = []
    for (let i = 0; i < x; i++) {
        indices.push(i)
    }
    return indices
}

/**
 * @param {arr} arr 
 * @param {shape} shape 
 * @param {axes} axes 
 * @returns transposed 1d array
 */
function transpose(arr, shape, axes) {
    const prevStride = []
    shape.forEach((_, i) => prevStride.push(shape.slice(i + 1).
        reduce((acc, v) => acc * v, 1)))
    const transposedShape = axes.map((axis) => shape[axis])
    const transposedStride = axes.map((axis) => prevStride[axis])

    const indices = transposedShape.map((x) => createIndices(x))
    const transposedArr = []

    for (let index of cartesianProduct(...indices)) {
        const transposedIdx = transposedStride.map((x, i) => index[i] * x)
            .reduce((acc, y) => acc + y, 0)
        transposedArr.push(arr[transposedIdx])
    }

    return { transposedArr, transposedShape }
}

/**
 * @param {data} data should be typed array
 * @returns simple ordinary array
 */
function fromTypedArrayToArray(data) {
    const container = []
    data.forEach((x) => container.push(x))
    return container
}


/**
 * This function helps to check whether other is array or tensor
 * @param {other} other 
 * @returns true if array or tensor otherwise false
 */
function isArrayOrNDArray(other) {
    if (Array.isArray(other) || (other instanceof NDArray)) {
        return true
    }
    return false
}


/**
 * Does matrix multiplication
 * @param {Array} m1 first matrix
 * @param {Array} m2 second matrix
 * @param {int} m1Rows number of rows of first matrix
 * @param {int} m1Cols number of columns of first matrix
 * @param {int} m2Cols number of columns of second matrix
 * @returns 1d array of shape(m1Rows, m2Cols)
 */
function matrixMul(m1, m2, m1Rows, m1Cols, m2Cols) {
    const indices = cartesianProduct(createIndices(m1Rows), createIndices(m2Cols))
    const result = new Array(m1Rows * m2Cols)
    for (let index of indices) {
        let sum = 0
        for (let k = 0; k < m1Cols; k++) {
            sum += m1[index[0] * m1Cols + k] * m2[k + k * (m2Cols - 1) + index[1]]
        }
        result[index[0] * m2Cols + index[1]] = sum
    }

    return result
}

/**
 * @param {Array} arr 
 * @returns strides array
 */
function calcStrides(arr) {
    const strides = []
    for (let i = 1; i <= arr.length; i++) {
        strides.push(arr.slice(i).reduce((acc, curr) => acc * curr, 1))
    }
    return strides
}

/**
 * 
 * @param {Array} ndArray 
 * @param {Array} shape 
 * @returns 2d array of matrices
 */
function getMatricesFromNDArray(ndArray, shape) {
    const matrices = []
    const newShape = shape.slice(0, shape.length - 2)
    const strides = calcStrides(shape)
    const newStrides = strides.slice(0, shape.length - 2)
    const idxArr = []
    for (let i = 0; i < newShape.length; i++) {
        idxArr.push(createIndices(newShape[i]))
    }

    const indices = cartesianProduct(...idxArr)

    for (const index of indices) {
        const mulStrideIdx = []
        newStrides.forEach((value, i) => mulStrideIdx.push(value * index[i]))
        const startIdx = mulStrideIdx.reduce((acc, cur) => acc + cur, 0)
        matrices.push(ndArray.slice(startIdx, startIdx + newStrides[newStrides.length - 1]))
    }

    return matrices
}