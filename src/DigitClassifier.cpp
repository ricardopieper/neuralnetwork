#include "DigitClassifier.hpp"
#include <iostream>
#include <random>
#include <climits>
#include <limits>
#include <chrono>
#include "fmincg.h"
#include "NNDefs.h"

using std::cout;

void DigitClassifier::logStr(std::string str){
    if (m_verbose){
        std::cout<<str<<std::endl;
    }
}

template<typename M>
void DigitClassifier::printSizeMtx(std::string str, M & x){
    if (m_verbose){

        logStr(str + " size is "+std::to_string(x.numRows()) + " "+ std::to_string(x.numCols()));


    }
}

Mtx DigitClassifier::sigmoid(Mtx  m){

    Mtx::IndexVariable i, j;
    Mtx G(m.numRows(), m.numCols());

    G(i,j) = 1.0 / (1.0 + Exp(-m(i, j)));

    return G;
}

Mtx DigitClassifier::sigmoidGradient(Mtx m){

    Mtx::IndexVariable i, j;

    Mtx s = sigmoid(m);

    Mtx g(m.numRows(), m.numCols());

    g(i,j) = s(i, j) * (1-s(i,j));

    return g;
}


Mtx DigitClassifier::randInitializeWeights(int lIn, int lOut){
//L_out, 1 + L_in

    double min = 0;
    double max = 1;

    std::uniform_real_distribution<> gen(min, max); // uniform, unbiased

    long time = std::chrono::system_clock::now()
            .time_since_epoch()
            .count();

    std::mt19937 rng(time);

    Mtx W(lOut, lIn);

    double epsilon = 0.12;

    for (int i = 1; i<= lOut; i++) {
        for (int j = 1; j<= lIn; j++) {
            W(i,j) = ((gen(rng) * 2) * epsilon) - epsilon;
        }
    }

    return W;

}

Mtx DigitClassifier::unroll(Mtx x){

    int _x = x.numRows() * x.numCols();

    Mtx UnrollX = MtxView(FSView(_x, 1, x.data(), _x));

    return UnrollX;
}



Mtx DigitClassifier::unroll(Mtx x, Mtx y){

    Mtx UnrollX = unroll(x);
    Mtx UnrollY = unroll(y);

    int rows = UnrollX.numRows() + UnrollY.numRows();

    Mtx Unrolled(rows, 1);

    Underscore<Mtx::IndexType>  _;

    Unrolled(_(1, UnrollX.numRows()),_) = UnrollX;
    Unrolled(_(UnrollX.numRows() + 1, rows),_) = UnrollY;

    return Unrolled;
}


Mtx DigitClassifier::reshape(Mtx  x, int rows, int cols){

    logSize("the matrix to reshape", x);

    log("Reshaping to " + std::to_string(rows) + " "+std::to_string(cols));


    assert(x.numRows() * x.numCols() >= (rows * cols));


    Mtx Reshaped = MtxView(FSView(rows,
                                  cols, x.data(), rows));

    return Reshaped;

}

template<typename M>
Mtx DigitClassifier::prependColumn(M  x, double val){

    Mtx y(x.numRows(), 1 + x.numCols());
    Underscore<typename M::IndexType>  _;
    y(_, _(2, 1 + x.numCols())) = x;
    y(_, _(1, 1)) = val;

    return y;
}

template<typename M>
Mtx DigitClassifier::prependRow(M  x, double val){

    Mtx y(1 + x.numRows(), x.numCols());
    Underscore<typename M::IndexType>  _;
    y(_(2, 1 + x.numRows()), _) = x;
    y(_(1,1), _) = val;

    return y;
}

Mtx DigitClassifier::mapLabels(Mtx labels){

    Mtx mapped(labels.numRows(), 10);
    Underscore<Mtx::IndexType>  _;

    for (int i=1; i<= labels.numRows(); i++){

        double value = labels(i, 1);

        switch ((int)value){
            case 1: mapped(i, _(1,10)) = 1,0,0,0,0,0,0,0,0,0; break;
            case 2: mapped(i, _(1,10)) = 0,1,0,0,0,0,0,0,0,0; break;
            case 3: mapped(i, _(1,10)) = 0,0,1,0,0,0,0,0,0,0; break;
            case 4: mapped(i, _(1,10)) = 0,0,0,1,0,0,0,0,0,0; break;
            case 5: mapped(i, _(1,10)) = 0,0,0,0,1,0,0,0,0,0; break;
            case 6: mapped(i, _(1,10)) = 0,0,0,0,0,1,0,0,0,0; break;
            case 7: mapped(i, _(1,10)) = 0,0,0,0,0,0,1,0,0,0; break;
            case 8: mapped(i, _(1,10)) = 0,0,0,0,0,0,0,1,0,0; break;
            case 9: mapped(i, _(1,10)) = 0,0,0,0,0,0,0,0,1,0; break;
            case 10: mapped(i, _(1,10)) = 0,0,0,0,0,0,0,0,0,1; break;
            default: break;
        }
    }

    return mapped;
}

double DigitClassifier::sumAll(Mtx x){
    auto vec = x.vectorView();
    double sum = 0;
    for (int i=1; i<=vec.length(); i++){
        sum += vec(i);
    }
    return sum;
}
Mtx DigitClassifier::predict(Mtx dataSet, Mtx Theta1,
                                Mtx Theta2){

    Mtx a1 = prependColumn(dataSet, 1);
    Mtx ffa2 = sigmoid(a1 * transpose(Theta1));
    Mtx a2 = prependColumn(ffa2, 1);


    Mtx output = sigmoid(a2 * transpose(Theta2));

   // std::cout<<output<<std::endl;

    Mtx result(output.numRows(), 1);

    for (int i=1; i <= output.numRows(); i++){

        double max = -1;

        int index = 0;

        for (int j=1; j <= output.numCols(); j++){

            double current = output(i,j);

            if (current > max) {
                max = current;
                index = j;
            }
        }
        result(i,1) = index;
    }

    return result;

}

Cost DigitClassifier::costFunction(Mtx nnParams,
                                   int inputLayerSize,
                                   int hiddenLayerSize,
                                   int numLabels,
                                   Mtx  dataSet,
                                   Mtx  labels,
                                   double lambda){

    //pause("check mem");
    m_verbose = false;

    Cost r;

    Mtx::IndexVariable i, j;
    Underscore<Mtx::IndexType>  _;

    Mtx Theta1 = reshape(nnParams,hiddenLayerSize,  inputLayerSize+1);
    int elemsForTheta1 = (inputLayerSize+1) * hiddenLayerSize;

    Mtx skipTheta1 = nnParams(_(elemsForTheta1+1, nnParams.numRows()), _);

    Mtx Theta2 = reshape(skipTheta1, numLabels, hiddenLayerSize + 1);
    double m = dataSet.numRows();

    Mtx a1 = prependColumn(dataSet, 1);
    Mtx ffa2 = sigmoid(a1 * transpose(Theta1));
    Mtx a2 = prependColumn(ffa2, 1);
    Mtx output = sigmoid(a2 * transpose(Theta2));

    Mtx firstPart(output.numRows(), output.numCols());
    Mtx secondPart(output.numRows(), output.numCols());


    //calculates the cost

    firstPart(i,j) = -(labels (i,j)) * Log(output(i,j));

    secondPart(i,j) = (1-labels (i,j)) * Log(1- output(i,j));

    double term  = (1/m)  * sumAll(firstPart - secondPart);

    //calculates the regularization term
    //skips the bias weight

    Mtx Theta1NoBias = Theta1(_, _(2, Theta1.numCols()));
    Mtx Theta2NoBias = Theta2(_, _(2, Theta2.numCols()));

    //square them
    Theta1NoBias(i,j) = Theta1NoBias(i,j) * Theta1NoBias(i,j);
    Theta2NoBias(i,j) = Theta2NoBias(i,j) * Theta2NoBias(i,j);

    double sum = sumAll(unroll(Theta1NoBias, Theta2NoBias));

    double regularization = sum * (lambda / (2*m));

    r.CostValue = term + regularization;

    std::cout<<"Calculated cost is ";
    printf("%.17f\n", r.CostValue);

    //now lets do the backprop

    Mtx Theta1_Gradient(Theta1.numRows(), Theta1.numCols());
    Mtx Theta2_Gradient(Theta2.numRows(), Theta2.numCols());

//backpropagation vectorized
//transposition shitfest

    Mtx z2 = transpose(Theta1 * transpose(a1));
    Mtx ffa2Sigmoid = sigmoid(z2);
    Mtx _ffa2 = prependColumn(ffa2Sigmoid, 1);
    Mtx z3 = transpose(Theta2 * transpose(_ffa2));
    Mtx ffa3 = sigmoid(z3);
    Mtx d3 = ffa3 - labels;
    Mtx d2 = transpose(transpose(Theta2) * transpose(d3));
    Mtx _d2sigmoid = prependColumn(sigmoidGradient(z2),1);
    d2(i,j) = d2(i,j) * _d2sigmoid(i,j);
    Mtx d2_skip = d2(_, _(2, d2.numCols()));
    Theta1_Gradient = Theta1_Gradient + (transpose(d2_skip) * a1);
    Theta2_Gradient = Theta2_Gradient + (transpose(d3) * _ffa2);

    auto Theta1SkipColumn1 = Theta1(_, _(2, Theta1.numCols()));
    auto Theta2SkipColumn1 = Theta2(_, _(2, Theta2.numCols()));

    Theta1_Gradient = (1/m) * Theta1_Gradient + (lambda/m) * prependColumn(Theta1SkipColumn1,0);

    Theta2_Gradient = (1/m) * Theta2_Gradient + (lambda/m) * prependColumn(Theta2SkipColumn1,0);


    r.Gradient = unroll(Theta1_Gradient, Theta2_Gradient);

   // std::cout<<"Last ";
    //printf("%.17f\n", r.Gradient(r.Gradient.numRows(), r.Gradient.numCols()));

    m_verbose = true;
    return r;

}


void DigitClassifier::Train(Mtx* X, Mtx* y, Mtx* Theta1, Mtx* Theta2, Mtx* RandomTheta1, Mtx* RandomTheta2) {
    Underscore<Mtx::IndexType>  _;
    int inputLayerSize = X->numCols();
    int hiddenLayerSize = 25;
    int numLabels = 10;

    Mtx unrolled = unroll(*Theta1, *Theta2);


    Mtx labels = mapLabels(*y);

  //  Cost c = costFunction(unrolled, inputLayerSize, hiddenLayerSize, numLabels,*X, labels, 0);

   // std::cout << "Cost: "<<c.CostValue<< std::endl;

    //add +1 for the bias unit
    Mtx initialTheta1 = *RandomTheta1;///randInitializeWeights(inputLayerSize + 1, hiddenLayerSize);
    Mtx initialTheta2 = *RandomTheta2;///randInitializeWeights(hiddenLayerSize + 1, numLabels);
    Mtx initialNNParams = unroll(initialTheta1, initialTheta2);

    double lambda = 0.1;

    m_verbose = true;

    std::function<Cost (Mtx)> cf =
            [this, inputLayerSize, hiddenLayerSize, numLabels,
                    X, labels, lambda](Mtx nnParams){


                //std::cout<<"Calling the cost function..."<<std::endl;
                // pause("Calling the cost function. Any key to continue.");
                return costFunction(nnParams, inputLayerSize, hiddenLayerSize,
                                    numLabels, *X, labels, lambda);

            };

    Mtx nn_params = fmincg(cf, initialNNParams, 100);

    log("Finished minimization");


    Mtx Trained_Theta1 = reshape(nn_params, hiddenLayerSize, inputLayerSize+1);

    int elemsForTheta1 = (inputLayerSize+1) * hiddenLayerSize;

    Mtx skipTheta1 = nn_params(_(elemsForTheta1+1, nn_params.numRows()), _);

    Mtx Trained_Theta2 = reshape(skipTheta1, numLabels, hiddenLayerSize + 1);


    log("Doing predictions in the train dataset...");

    Mtx predictionTrain = predict(*X, Trained_Theta1, Trained_Theta2);

    std::cout<<predictionTrain.numRows()<<std::endl;

    log("Predictions done. Verifying..");

    int correct = 0;

    for(int i=1; i <= predictionTrain.numRows(); i++) {

        if (predictionTrain(i,1) == (*y)(i,1)){
            correct++;
        }
    }

    log("Predictions are done. "+std::to_string(correct)+" predictions are correct.");


}