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

// Return double array of weights of length [num_rows*num_cols]
double *parse_weights_csv(const std::string filename, const int num_rows,
                          const int num_cols) {
  std::cout << "Loading " << filename << "...." << std::endl;
  std::ifstream weight_file(filename.c_str(), std::ios_base::in);
  double *weights = (double *)_malloc(sizeof(double) * num_rows * num_cols);

  std::string buffer;
  if (weight_file) {
    for (int i = 0; i < num_rows; i++) {
      double *d = &weights[i * num_cols];
      for (int j = 0; j < num_cols; j++) {
        double f;
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
int *parse_ids_csv(const std::string filename, const int num_rows) {
  std::cout << "Loading " << filename << "...." << std::endl;
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    std::cout << "Unable to open " << filename << std::endl;
    exit(1);
  }
  int *ids = (int *)mkl_malloc(sizeof(int) * num_rows, 64);

  std::string line;
  int i = 0;
  while (getline(in, line)) {
    ids[i++] = std::stoi(line);
  }
  in.close();
  return ids;
}
