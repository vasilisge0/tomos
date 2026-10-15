#ifndef TOMOS_DIMENSIONS_HPP
#define TOMOS_DIMENSIONS_HPP

#include <stdexcept>

namespace tomos {

template <int len> struct dim;

template<> struct dim<2> {
  size_t nrows = 0;
  size_t ncols = 0;

  dim(size_t nr, size_t nc) {
    nrows = nr;
    ncols = nc;
  }

  size_t operator[](size_t idx) {
    switch (idx) {
    case 0:
      return nrows;
    case 1:
      return ncols;
    default:
      throw std::invalid_argument("index should be either 0 or 1");
    }
  }

  bool operator==(dim<2> rhs) {
    return (nrows == rhs.nrows) && (ncols == rhs.ncols) ? true : false;
  } // end of operator==
  
  bool operator!=(dim<2> rhs) {
    return (nrows != rhs.nrows) || (ncols != rhs.ncols) ? true : false;
  } // end of operator!=
};

} // end of namespace rls

#endif // TOMOS_DIMENSIONS_HPP
