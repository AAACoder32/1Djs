console.log("Hello from Tensor.js")
class Tensor {
    /**
     * Takes argument as nd array
     * @param {Array} data 
     * @param {String} dtype 
     */
    constructor(data, dtype = "float32") {
        this.dtype = dtype
        this.shape = this.#getShape(data)
        this.ndim = this.shape.length
        this.data = this.#initData(data, dtype)
    }

    #initData(data, dtype) {
        if (!Array.isArray(data)) {
            throw new Array("Tensor expects data arg as an array")
        }
        switch (dtype) {
            case "int32":
                return new Int32Array(data.flat(this.ndim))
            case "float32":
                return new Float32Array(data.flat(this.ndim))
            default:
                throw new Error("Unsupported data type")
        }
    }

    /**
     * @param {data} data array or multi dimensional array
     * @returns shape of array
     */
    #getShape(data) {

        if (checkIsErrorInDataShape(data)) {
            console.error("There is an element is less or more in your ndarray")
            return
        }

        const shape = []
        getRecursively(data, shape)

        function getRecursively(data) {
            if (!Array.isArray(data)) {
                return
            }
            shape.push(data.length)
            data = data[0]
            getRecursively(data)
        }

        return shape
    }

    /**
     * @param {shape} shape should be 1d array
     * @returns new reshaped tensor
     */
    reshape(shape) {
        if (!Array.isArray(shape)) {
            throw new Error("Shape should 1d array")
        }
        if (shape.reduce((a, b) => a * b) !== this.data.length) {
            throw new Error("Shape is not compatible with array length");
        }
        return new Tensor(makeMultiDimentional(shape, fromTypedArrayToArray(this.data)), this.dtype)
    }

    /**
     * @param {axes} axes should be 1d array of positive integer (0,1,3,...) 
     * don't give any args
     * @returns new transposed tensor
     */
    transpose(axes = undefined) {
        if (!Array.isArray(axes) && axes !== undefined) {
            throw new Error("Shape should 1d array or don't give any args")
        }

        if (axes === undefined) {
            if (this.ndim === 1) {
                return this
            }
            axes = createIndices(this.shape.length).reverse()
            const obj = this.#_transpose(this.data, this.shape, axes)
            return new Tensor(makeMultiDimentional(obj.transposedShape,
                obj.transposedArr), this.dtype)
        }

        if ((axes.length === this.ndim)) {
            let temp = Array.from(axes)
            temp.sort()
            this.shape.forEach((_, i) => {
                if (temp[i] !== i) {
                    throw new Error("Kya kar raha hai be!")
                }
            })
        } else {
            throw new Error("Length of axes array should be same as shape length. Samajha!")
        }

        if (this.ndim === 1) {
            return this
        }

        const obj = this.#_transpose(this.data, this.shape, axes)
        return new Tensor(makeMultiDimentional(obj.transposedShape,
            obj.transposedArr), this.dtype)
    }


    /**
     * To print tensor in readable manner
     */
    print() {
        printArray(this.data, this.shape);
    }


    /**
     * other should be tensor object or same dim type array or broadcastable
     * @param {other} other 
     * @returns array
     */
    add(other) {
        const isValid = isArrayOrTensor(other)
        if (!isValid) {
            throw new Error("Other should be array or tensor obj.")
        }

        if (Array.isArray(other)) {
            other = new Tensor(other)
        }

        let isTrue = false
        if (this.ndim === other.ndim) {
            for (let i = 0; i < this.ndim; i++) {
                if (this.shape[i] === other.shape[i]) {
                    isTrue = true
                } else {
                    isTrue = false
                }
            }
        }
        if (isTrue) {
            const res = this.data.map((x, i) => x + other.data[i])
            return new Tensor(makeMultiDimentional(this.shape, fromTypedArrayToArray(res)), this.dtype)
        }
        const broadcasted = this.#broadcastTo(other)
        const res = broadcasted.self.map((x, i) => x + broadcasted.other[i])
        console.log(broadcasted)
        return new Tensor(makeMultiDimentional(broadcasted.shape, res),
            broadcasted.self.dtype)
    }

    /**
      * other should be tensor object or same dim type array or broadcastable
      * @param {other} other 
      * @returns array
      */
    subtr(other) {
        const isValid = isArrayOrTensor(other)
        if (!isValid) {
            throw new Error("Other should be array or tensor obj.")
        }

        if (Array.isArray(other)) {
            other = new Tensor(other)
        }

        let isTrue = false
        if (this.ndim === other.ndim) {
            for (let i = 0; i < this.ndim; i++) {
                if (this.shape[i] === other.shape[i]) {
                    isTrue = true
                } else {
                    isTrue = false
                }
            }
        }
        if (isTrue) {
            const res = this.data.map((x, i) => x - other.data[i])
            return new Tensor(makeMultiDimentional(this.shape, fromTypedArrayToArray(res)), this.dtype)
        }
        const broadcasted = this.#broadcastTo(other)
        const res = broadcasted.self.map((x, i) => x - broadcasted.other[i])
        return new Tensor(makeMultiDimentional(broadcasted.shape, res),
            broadcasted.self.dtype)
    }

    /**
    * other should be tensor object or same dim type array or broadcastable
    * @param {other} other 
    * @returns array
    */
    eWiseMul(other) {
        const isValid = isArrayOrTensor(other)
        if (!isValid) {
            throw new Error("Other should be array or tensor obj.")
        }

        if (Array.isArray(other)) {
            other = new Tensor(other)
        }

        let isTrue = false
        if (this.ndim === other.ndim) {
            for (let i = 0; i < this.ndim; i++) {
                if (this.shape[i] === other.shape[i]) {
                    isTrue = true
                } else {
                    isTrue = false
                }
            }
        }
        if (isTrue) {
            const res = this.data.map((x, i) => x * other.data[i])
            return new Tensor(makeMultiDimentional(this.shape, fromTypedArrayToArray(res)), this.dtype)
        }
        const broadcasted = this.#broadcastTo(other)
        let res = broadcasted.self.map((x, i) => x * broadcasted.other[i])
        return new Tensor(makeMultiDimentional(broadcasted.shape, res),
            broadcasted.self.dtype)
    }

    /**
    * other should be tensor object or same dim type array or broadcastable
    * @param {other} other 
    * @returns array
    */
    divide(other) {
        const isValid = isArrayOrTensor(other)
        if (!isValid) {
            throw new Error("Other should be array or tensor obj.")
        }

        if (Array.isArray(other)) {
            other = new Tensor(other)
        }

        let isTrue = false
        if (this.ndim === other.ndim) {
            for (let i = 0; i < this.ndim; i++) {
                if (this.shape[i] === other.shape[i]) {
                    isTrue = true
                } else {
                    isTrue = false
                }
            }
        }
        if (isTrue) {
            const res = this.data.map((x, i) => x / other.data[i])
            return new Tensor(makeMultiDimentional(this.shape, fromTypedArrayToArray(res)), this.dtype)
        }
        const broadcasted = this.#broadcastTo(other)
        const res = broadcasted.self.map((x, i) => x / broadcasted.other[i])
        return new Tensor(makeMultiDimentional(broadcasted.shape, res),
            broadcasted.self.dtype)
    }

    /**
     * Other should be object of tensor or array
     * @param {other} other
     * @returns object of arrays 
     */
    #broadcastTo(other) {

        if (!Array.isArray(other) && !(other instanceof Tensor)) {
            throw new Error("Other should be object of tensor or array")
        }

        if (Array.isArray(other)) {
            other = new Tensor(other)
        }

        const resultShape = broadcastShapes(this.shape, other.shape);
        return {
            "self": this.#fillArray(resultShape, this.data),
            "other": this.#fillArray(resultShape, other.data),
            "shape": resultShape
        }
    }

    /**
     * This methods takes 1d array, shape and returns modified array according to shape
     * @param {shape} shape 
     * @param {arr} arr 
     * @returns result
     */
    #fillArray(shape, arr) {
        const result = [];
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
     * 
     * @param {arr} arr should be 1d array of multi dimension array or 1d,2d ,3d etc.
     * @param {shape} shape should be 1d array
     * @param {axes} axes should be 1d array
     * @returns transposed 1d array of multi dimension array or 1d,2d ,3d etc
     *  with its shape.
     */
    #_transpose(arr, shape, axes) {
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
     * @returns new tensor with tanh operation
     */
    tanh() {
        const arr = fromTypedArrayToArray(this.data)
        const v = new Tensor(arr, this.dtype)
        v.shape = this.shape
        v.data = v.data.map((x) => Math.tanh(x))
        return v
    }


    /**
     * @returns new tensor with e^x operation
     */
    exp() {
        const arr = fromTypedArrayToArray(this.data)
        const v = new Tensor(arr, this.dtype)
        v.shape = this.shape
        v.data = v.data.map((x) => Math.exp(x))
        return v
    }

    /**
     * add scalar to each elements of tensor
     * @param {scalar} scalar 
     * @returns tensor
     */
    addScalar(scalar) {
        const arr = fromTypedArrayToArray(this.data)
        const v = new Tensor(arr, this.dtype)
        v.shape = this.shape
        v.data = v.data.map((x) => x + scalar)
        return v
    }

    /**
    * subtract scalar from each elements of tensor
    * @param {scalar} scalar 
    * @returns tensor
    */
    subtrScalar(scalar) {
        const arr = fromTypedArrayToArray(this.data)
        const v = new Tensor(arr, this.dtype)
        v.shape = this.shape
        v.data = v.data.map((x) => x - scalar)
        return v
    }

    /**
    * multiply scalar to each elements of tensor
    * @param {scalar} scalar 
    * @returns tensor
    */
    mulScalar(scalar) {
        const arr = fromTypedArrayToArray(this.data)
        const v = new Tensor(arr, this.dtype)
        v.shape = this.shape
        v.data = v.data.map((x) => x * scalar)
        return v
    }

    /**
     * Divide by scalar to each elements of tensor
     * @param {scalar} scalar 
     * @returns tensor
     */
    divScalar(scalar) {
        const arr = fromTypedArrayToArray(this.data)
        const v = new Tensor(arr, this.dtype)
        v.shape = this.shape
        v.data = v.data.map((x) => x / scalar)
        return v
    }

    /**
     * 
     * @param {axis} axis should be integer from 0 to ndim-1
     * @returns sum over a given axis
     */
    contract(axis) {

        if (axis === undefined) {
            const res = this.data.reduce((acc, cur) => acc + cur, 0)
            return new Tensor([res], this.dtype)
        }

        if (axis !== Number(axis)) {
            throw new Error("Axis should be integer")
        }

        if (axis < 0 || axis > this.ndim - 1) {
            throw new Error("Axis should be in range 0 to ndim-1.")
        }

        const shape = this.shape
        const strides = []
        shape.forEach((_, i) => strides.push(shape.slice(i + 1).
            reduce((acc, v) => acc * v, 1))) // Stride calculation
        const shapeNew = shape.filter((_, i) => i !== axis)
        const stridesNew = []
        shapeNew.forEach((_, i) => stridesNew.push(shapeNew.slice(i + 1).
            reduce((acc, v) => acc * v, 1))) // Stride calculation
        // Filter stride excluding shape[axis] stride
        const stridesOld = strides.filter((_, i) => i !== axis)
        // Axis stride for convenient
        const strideAxis = strides[axis]
        // Cartesian product of all possible indices
        const indices = shapeNew.map((x) => createIndices(x))

        const resultArr = new Array(shapeNew.reduce((acc, x) => acc * x, 1))
        for (let index of cartesianProduct(...indices)) {
            let idxNew = 0
            index.forEach((value, i) => {
                idxNew += value * stridesNew[i]
            })

            let idxOld = 0
            index.forEach((value, i) => {
                idxOld += value * stridesOld[i]
            })

            resultArr[idxNew] = createIndices(shape[axis])
                .map((x) => this.data[idxOld + x * strideAxis])
                .reduce((acc, v) => acc + v, 0)
        }

        return new Tensor(makeMultiDimentional(shapeNew, resultArr), this.dtype)
    }

    /**
     * @param {Tensor|Array} other 
     * @returns Tensor of matrix multiplication
     */
    matmul(other) {
        if (!Array.isArray(other) && !(other instanceof Tensor)) {
            throw new Error("Other should be array or tensor")
        }

        if (Array.isArray(other)) {
            other = new Tensor(other)
        }

        if (this.shape[this.ndim - 1] === other.shape[other.ndim - 2]) {
            if (this.ndim === 2 && other.ndim === 2) {
                const result = matrixMul(this.data, other.data, this.shape[0], this.shape[1], other.shape[1])
                const v = new Tensor(result, this.dtype)
                v.ndim = 2
                v.shape = [this.shape[0], other.shape[1]]
                return v
            }

            if (this.ndim === other.ndim) {
                let isAll = false
                for (let i = 0; i < this.ndim - 2; i++) {
                    if (this.shape[i] === other.shape[i]) {
                        isAll = true
                    } else {
                        isAll = false
                        break
                    }
                }
                if (isAll) {
                    const ms1 = getMatricesFromNDArray(this.data, this.shape)
                    const ms2 = getMatricesFromNDArray(other.data, other.shape)
                    const result = []
                    ms1.forEach((value, i) => {
                        result.push(matrixMul(value, ms2[i],
                            this.shape[this.ndim - 2], this.shape[this.ndim - 1],
                            other.shape[other.ndim - 1]))
                    })
                    const v = new Tensor(result.flat(), this.dtype)
                    v.ndim = this.ndim
                    v.shape = this.shape.slice(0, this.ndim - 2)
                    v.shape.push(this.shape[this.ndim - 2], other.shape[other.ndim - 1])
                    return v
                }
            }

            const s1 = this.shape.slice(0, (this.ndim - 2))
            const s2 = other.shape.slice(0, (other.ndim - 2))
            const broadcastedShape = broadcastShapes(s1, s2)
            // console.log(broadcastedShape[1])
            const shapeT1 = []
            shapeT1.push(...broadcastedShape)
            shapeT1.push(this.shape[this.ndim - 2], this.shape[this.ndim - 1])
            const shapeT2 = []
            shapeT2.push(...broadcastedShape)
            shapeT2.push(other.shape[other.ndim - 2], other.shape[this.ndim - 1])

            const ms1 = getMatricesFromNDArray(this.broadcastTo(shapeT1).data, shapeT1)
            const ms2 = getMatricesFromNDArray(other.broadcastTo(shapeT2).data, shapeT2)
            console.log(ms1, ms2)
            const result = []
            ms1.forEach((value, i) => {
                result.push(matrixMul(value, ms2[i],
                    shapeT1[this.ndim - 2], shapeT1[this.ndim - 1],
                    shapeT2[other.ndim - 1]))
            })
            const v = new Tensor(result.flat(), this.dtype)
            v.ndim = shapeT1.length
            v.shape = shapeT1.slice(0, v.ndim - 2)
            v.shape.push(shapeT1[v.ndim - 2], shapeT2[v.ndim - 1])
            return v
        }
        else {
            throw new Error("Shape is not compatible for matmul")
        }
    }

    broadcastTo(targetShape) {
        const inputShape = this.shape;
        const inputData = this.data;
        const outputData = new Array(targetShape.reduce((a, b) => a * b, 1));
        const outputStrides = calcStrides(targetShape);

        let inputIndex = 0;
        for (let i = 0; i < outputData.length; i++) {
            const coords = getCoordsFromIndex(i, targetShape, outputStrides);
            console.log(`Coords: ${coords}`);
            const broadcastedCoords = coords.slice(-inputShape.length);
            console.log(`Broadcasted Coords: ${broadcastedCoords}`);
            inputIndex = calculateInputIndex(broadcastedCoords, inputShape, calcStrides(inputShape));
            console.log(`Input Index: ${inputIndex}`);
            outputData[i] = inputData[inputIndex];
        }

        const v = new Tensor(outputData, this.dtype);
        v.shape = targetShape;
        v.ndim = targetShape.length;
        return v;
    }


}