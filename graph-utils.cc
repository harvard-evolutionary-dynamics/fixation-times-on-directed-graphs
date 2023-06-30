
#include <vector>
#include <utility>
#include "eigen/Eigen/Sparse"
#include "eigen/Eigen/Dense"
#include <vector>
#include <iostream>
 
// using SpMat = Eigen::SparseMatrix<double>;
using SpMat = Eigen::MatrixXd;
// using MatrixXD = Matrix<double, Dynamic, Dynamic>;

using Graph = std::vector<std::pair<size_t, size_t>>;
using Degrees = std::vector<size_t>;

int vec2num(const std::vector<size_t>& vec) {
  const int N = vec.size();

  int acc = 0;
  for (size_t i = 0; i < N; ++i) {
    acc += (1 << i) * vec[N-i-1];
  }
  return acc;
}

Degrees g2degs(const Graph& G) {
  const auto v = G[0].first;
  const auto e = G[0].second;

  Degrees degs(v);
  for (int i = 0; i < v; i++) degs[i] = 0;
  for (size_t i = 1; i < G.size(); ++i) {
    const auto& ed = G[i];
    ++degs[ed.first];
    ++degs[ed.second];
    if (ed.first == ed.second) --degs[ed.first];
  }
  return degs;
}

bool nextbin(std::vector<size_t>* bits) {
  bool found = false;
  for (int i = bits->size()-1; i >= 0; --i) {
    if ((*bits)[i] == 0) {
      found = true;
      (*bits)[i] = 1;
      for (int j = i+1; j < bits->size(); ++j) {
        (*bits)[j] = 0;
      }
      break;
    }
  }
  return found;
}

SpMat g2mat(const Graph& G, const Degrees& degs, const double r) {
  const auto v = G[0].first;
  const auto e = G[0].second;
  const size_t states = 1 << v;
  SpMat Mprob{states, states};
  for (int i = 0; i < states; i++) {
    for (int j = 0; j < states; j++) {
      Mprob(i, j) = 0;
    }
  }
  std::vector<size_t> powers;

  for (size_t k = 0; k < v; ++k) {
    powers.push_back(1 << (v-1-k));
  }

  std::vector<size_t> vecst(v);
  for (int i = 0; i < v; i++) vecst[i] = 0;
  do {
    std::vector<double> mut;
    for (size_t k = 0; k < v; ++k) {
      mut.push_back(vecst[k] == 1 ? r : 1);
    }
    const auto numst = vec2num(vecst);
    double totfit = 0;
    for (const auto& x : mut) totfit += x;

    for (size_t i = 1; i < G.size(); ++i) {
      const auto& ed = G[i];
      const auto k = ed.first;
      const auto l = ed.second;
      if (vecst[k] ==  vecst[l]) continue;

      size_t stnew;
      if (vecst[k] == 0) stnew = numst-powers[l];
      else stnew=numst+powers[l];

      Mprob(numst,stnew)+=mut[k]/((double)(totfit*degs[k]));
      Mprob(numst,numst)-=mut[k]/((double)(totfit*degs[k]));
      
      if (vecst[l]==0) stnew=numst-powers[k];
      else stnew=numst+powers[k];

      Mprob(numst,stnew)+=mut[l]/((double)(totfit*degs[l]));
      Mprob(numst,numst)-=mut[l]/((double)(totfit*degs[l]));
    }
  } while (nextbin(&vecst));
  Mprob(0,0)=1;
  Mprob(states-1,states-1)=1;
  
  return Mprob;
}

std::vector<double> fprob(const SpMat& mat) {
  const auto& size= mat.rows();
  Eigen::VectorXd rhs(size);
  for (int i = 0; i < size; i++) rhs(i)=0;
  rhs(size-1)=1;
  const auto& fprobsol = mat.colPivHouseholderQr().solve(rhs);
  
  std::vector<double> retprobs;
  int ind = 1;
  while (ind<size) {
    retprobs.push_back(fprobsol(ind));
    ind *= 2;
  }
  std::reverse(retprobs.begin(), retprobs.end());
  return retprobs;
}

std::vector<double> atime(const SpMat& mat) {
    const auto& size= mat.rows();
    Eigen::VectorXd rhstime(size);
    for (int i = 0; i < size; i++) rhstime(i)=-1;
    rhstime(0)=0;
    rhstime(size-1)=0;
    const auto& ftimesol = mat.colPivHouseholderQr().solve(rhstime);
    
    std::vector<double> rettimes;
    int ind=1;
    while (ind<size) {
      rettimes.push_back(ftimesol(ind));
      ind*=2;
    }
    std::reverse(rettimes.begin(), rettimes.end());
    return rettimes;
}

std::vector<double> cftime(const SpMat& mat) {
  const auto& size= mat.rows();
  std::cout << mat << std::endl;
  
  Eigen::VectorXd rhs(size);
  for (int i = 0; i < size; i++) rhs(i)=0;
  rhs(size-1)=1;

  std::cout << rhs << std::endl;
  // const auto& fprobsol = mat.colPivHouseholderQr().solve(rhs);
  const auto& fprobsol = mat.ldlt().solve(rhs);

  SpMat cmat = mat;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (fprobsol(i) != 0) {
        cmat(i,j)=cmat(i,j)*fprobsol(j)/((double)(fprobsol(i)));
      }
    }
  }
  return atime(cmat);
}

// auto solvegraph(G,r) {
//   n,e=G[0] # stores the number of nodes and edges of G
//   degs=g2degs(G) # a list counting how many neighbors each node has
//   mat=g2mat(G,degs,r) # a transition matrix of the underlying Markov chain
// 
//   const auto fps=fprob(mat);
//   const auto ats=atime(mat);
//   const auto cfts=cftime(mat);
// 
//   // results=[fps,ats,cfts]
//   // return [round(sum(x)/float(n),6) for x in results]
//   }

int main() {
  const Graph G = {
    {4,6},{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}
  };
  const auto n =G[0].first;
  const auto e = G[0].second;
  const double r = 1.0;
  const auto degs=g2degs(G);
  const auto mat=g2mat(G,degs,r); 
  const auto cfts = cftime(mat);
  for (int i = 0; i < n; i++) {
    std::cout << i << ": " << cfts[i] << std::endl;
  }
  return 0;
}