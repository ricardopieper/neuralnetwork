#ifndef _NNDEFS_H_
#define _NNDEFS_H_

#include <flens/flens.cxx>

using namespace flens;

typedef FullStorage<double, ColMajor>  FS;
typedef FullStorageView<double, ColMajor>  FSView;
typedef GeMatrix<FS> Mtx;
typedef GeMatrix<FSView> MtxView;

typedef FullStorage<char, ColMajor>  ByteFS;
typedef FullStorageView<char, ColMajor> ByteFSView;
typedef GeMatrix<ByteFS> ByteMtx;
typedef GeMatrix<ByteFSView> ByteMtxView;

typedef FullStorage<int, ColMajor>  IntFS;
typedef FullStorageView<int, ColMajor> IntFSView;
typedef GeMatrix<IntFS> IntMtx;
typedef GeMatrix<IntFSView> IntMtxView;
 

class Cost {
public:
  double CostValue;
  Mtx Gradient;
};



#endif // _NNDEFS_H_
