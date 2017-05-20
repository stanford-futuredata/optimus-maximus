//
//  parser.cpp
//  SimDex
//
//  Created by Geet Sethi on 10/24/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#include "parser.hpp"
#include "utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <mkl.h>

// Return array of weights of type T (either float or double) and of length
// [num_rows*num_cols]
template <typename T>
T *parse_weights_csv(const std::string filename, const int num_rows,
                     const int num_cols) {
  std::cout << "Loading " << filename << "...." << std::endl;
  std::ifstream weight_file(filename.c_str(), std::ios_base::in);
  T *weights = (T *)_malloc(sizeof(T) * num_rows * num_cols);

  std::string buffer;
  if (weight_file) {
    for (int i = 0; i < num_rows; i++) {
      T *d = &weights[i * num_cols];
      for (int j = 0; j < num_cols; j++) {
        T f;
        weight_file >> f;
        if (j != num_cols - 1) {
          std::getline(weight_file, buffer, ',');
        }
        d[j] = f;
      }
      std::getline(weight_file, buffer);
    }
  }
  weight_file.close();
  return weights;
}

// Assume user ids are consecutively numbered, with no gaps
uint32_t *parse_ids_csv(const std::string filename, const int num_rows) {
  std::cout << "Loading " << filename << "...." << std::endl;
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    std::cout << "Unable to open " << filename << std::endl;
    exit(1);
  }
  uint32_t *ids = (uint32_t *)mkl_malloc(sizeof(uint32_t) * num_rows, 64);

  std::string line;
  uint32_t i = 0;
  while (getline(in, line)) {
    ids[i++] = std::stoi(line);
  }
  in.close();
  return ids;
}

template float *parse_weights_csv<float>(const std::string filename,
                                         const int num_rows,
                                         const int num_cols);
template double *parse_weights_csv<double>(const std::string filename,
                                           const int num_rows,
                                           const int num_cols);
