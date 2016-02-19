#include "NNDefs.h"
#define log(str) if (m_verbose) logStr(str)
#define logSize(str, mtx) if (m_verbose) printSizeMtx(str, mtx)

using namespace std;

class DigitClassifier{
private:

    bool m_verbose = false;

    void logStr(std::string str);

    template<typename M>
    void printSizeMtx(std::string str, M & x);
    Mtx sigmoid(Mtx m);
    Mtx sigmoidGradient(Mtx m);
    Mtx randInitializeWeights(int lIn, int lOut);
    Mtx unroll(Mtx x, Mtx y);
    Mtx unroll(Mtx x);
    Mtx reshape(Mtx x, int rows, int cols);
    Cost costFunction(Mtx nnParams,
                      int inputLayerSize,
                      int hiddenLayerSize,
                      int numLabels,
                      Mtx dataSet,
                      Mtx labels,
                      double lambda);

    template<typename M>
    Mtx prependRow(M x, double val);

    template<typename M>
    Mtx prependColumn(M x, double val);

    Mtx mapLabels(Mtx labels);
    double sumAll(Mtx x);

    Mtx predict(Mtx dataSet, Mtx Theta1,
                   Mtx Theta2);


public:
  void Train(Mtx* X, Mtx* y, Mtx* Theta1, Mtx* Theta2, Mtx* RandomTheta1, Mtx* RandomTheta2);
};
