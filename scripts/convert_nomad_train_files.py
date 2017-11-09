#!/usr/bin/env python

from scipy.sparse import csr_matrix
import sys
import os
import random
import struct

random.seed(12345)


def main():
    if len(sys.argv) < 3:
        print "usage: %s [train_filename] [test_filename] [output_path]" % sys.argv[
            0]
        exit(1)

    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    output_path = sys.argv[3]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    user_ids = set()
    item_ids = set()

    # index users and items
    train_file = open(train_filename)
    test_file = open(test_filename)

    for index, line in enumerate(train_file):

        if index % 1000000 == 0:
            print "1st pass training:", index

        tokens = line.split("\t")

        user_ids.add(tokens[0])
        item_ids.add(tokens[1])

        #if index > 200:
        #    break

    for index, line in enumerate(test_file):

        if index % 1000000 == 0:
            print "1st pass test:", index

        tokens = line.split("\t")

        user_ids.add(tokens[0])
        item_ids.add(tokens[1])

        #if index > 200:
        #    break

    user_id_list = list(user_ids)
    item_id_list = list(item_ids)
    random.shuffle(user_id_list)
    random.shuffle(item_id_list)

    user_indexer = {key: value for value, key in enumerate(user_id_list)}
    item_indexer = {key: value for value, key in enumerate(item_id_list)}

    # now parse the data
    train_user_indices = list()
    train_item_indices = list()
    train_values = list()

    train_file.seek(0)
    for index, line in enumerate(train_file):

        if index % 1000000 == 0:
            print "2nd pass training:", index

        tokens = line.split("\t")

        train_user_indices.append(user_indexer[tokens[0]])
        train_item_indices.append(item_indexer[tokens[1]])
        train_values.append(float(tokens[2]))

        #if index > 200:
        #    break

    #print user_indices
    #print item_indices
    #print values

    print "form training csr matrix"
    train_mat = csr_matrix(
        (train_values, (train_user_indices, train_item_indices)),
        shape=(len(user_indexer), len(item_indexer)))

    print "calculate size of rows"
    train_row_sizes = train_mat.indptr[1:] - train_mat.indptr[:-1]

    #print user_indexer
    #print len(user_indexer)
    #print train_row_sizes

    #print train_mat
    #print mat.indices
    #print mat.data

    print "write train binary file"
    ofile = open(output_path + "/train.dat", "wb")
    ofile.write(
        struct.pack("=iiii", 1211216,
                    len(user_indexer), len(item_indexer), train_mat.getnnz()))
    ofile.write(struct.pack("=%si" % len(train_row_sizes), *train_row_sizes))
    ofile.write(
        struct.pack("=%si" % len(train_mat.indices), *train_mat.indices))
    ofile.write(struct.pack("=%sd" % len(train_mat.data), *train_mat.data))
    ofile.close()

    test_user_indices = list()
    test_item_indices = list()
    test_values = list()

    test_file.seek(0)
    for index, line in enumerate(test_file):

        if index % 1000000 == 0:
            print "2nd pass test:", index

        tokens = line.split("\t")

        test_user_indices.append(user_indexer[tokens[0]])
        test_item_indices.append(item_indexer[tokens[1]])
        test_values.append(float(tokens[2]))

        #if index > 200:
        #    break

    #print user_indices
    #print item_indices
    #print values

    print "form test csr matrix"
    test_mat = csr_matrix(
        (test_values, (test_user_indices, test_item_indices)),
        shape=(len(user_indexer), len(item_indexer)))

    print "calculate size of rows"
    test_row_sizes = test_mat.indptr[1:] - test_mat.indptr[:-1]

    #print row_sizes
    #print mat.indices
    #print mat.data

    print "write test binary file"
    ofile = open(output_path + "/test.dat", "wb")
    ofile.write(
        struct.pack("=iiii", 1211216,
                    len(user_indexer), len(item_indexer), test_mat.getnnz()))
    ofile.write(struct.pack("=%si" % len(test_row_sizes), *test_row_sizes))
    ofile.write(struct.pack("=%si" % len(test_mat.indices), *test_mat.indices))
    ofile.write(struct.pack("=%sd" % len(test_mat.data), *test_mat.data))
    ofile.close()

    print "write user index mappings"
    ofile = open(output_path + "/user_ids.txt", "w")
    for user_id in user_id_list:
        ofile.write("%s\n" % user_id)
    ofile.close()

    print "write item index mappings"
    ofile = open(output_path + "/item_ids.txt", "w")
    for item_id in item_id_list:
        ofile.write("%s\n" % item_id)
    ofile.close()

    train_file.close()
    test_file.close()


if __name__ == '__main__':
    main()
