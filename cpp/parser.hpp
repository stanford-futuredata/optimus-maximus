//
//  parser.hpp
//  fomo_preproc
//
//  Created by Geet Sethi on 10/24/16.
//  Copyright © 2016 Geet Sethi. All rights reserved.
//

#ifndef parser_hpp
#define parser_hpp

#include "types.hpp"
#include <string>
#include <stdio.h>

float *parse_weights_csv(const std::string filename, const size_t num_rows,
                         const size_t num_cols);
int *parse_ids_csv(const std::string filename, const size_t num_rows);

#endif /* parser_hpp */
