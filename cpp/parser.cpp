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
#include <string.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <vector>

#include <boost/tokenizer.hpp>

#include <mkl.h>

float *parse_weights_csv(const std::string filename, const int num_rows,
                         const int num_cols) {
  std::cout << "Loading " << filename << "...." << std::endl;
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    std::cout << "Unable to open " << filename << std::endl;
    exit(1);
  }
  float *weights = (float *)_malloc(sizeof(float) * num_rows * num_cols);

  std::string line;
  int i = 0;
  while (getline(in, line)) {
    boost::tokenizer<boost::escaped_list_separator<char> > tk(
        line, boost::escaped_list_separator<char>('\\', ',', '\"'));
    for (boost::tokenizer<boost::escaped_list_separator<char> >::iterator it(
             tk.begin());
         it != tk.end(); ++it) {
      weights[i++] = std::stof(*it);
    }
  }
  in.close();
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
