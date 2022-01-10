#include "knn.h"

// Makefile included in starter:
//    To compile:               make
//    To decompress dataset:    make datasets
//
// Example of running validation (K = 3, 8 processes):
//    ./classifier 3 datasets/training_data.bin datasets/testing_data.bin 8

/*****************************************************************************/
/* This file should only contain code for the parent process. Any code for   */
/*      the child process should go in `knn.c`. You've been warned!          */
/*****************************************************************************/

/**
 * main() takes in 4 command line arguments:
 *   - K:  The K value for kNN
 *   - training_data: A binary file containing training image / label data
 *   - testing_data: A binary file containing testing image / label data
 *   - num_procs: The number of processes to be used in validation
 * 
 * You need to do the following:
 *   - Parse the command line arguments, call `load_dataset()` appropriately.
 *   - Create the pipes to communicate to and from children
 *   - Fork and create children, close ends of pipes as needed
 *   - All child processes should call `child_handler()`, and exit after.
 *   - Parent distributes the testing set among children by writing:
 *        (1) start_idx: The index of the image the child should start at
 *        (2)    N:      Number of images to process (starting at start_idx)
 *     Each child should gets N = ceil(test_set_size / num_procs) images
 *      (The last child might get fewer if the numbers doesn't divide perfectly)
 *   - Parent waits for children to exit, reads results through pipes and keeps
 *      the total sum.
 *   - Print out (only) one integer to stdout representing the number of test 
 *      images that were correctly classified by all children.
 *   - Free all the data allocated and exit.
 */
int main(int argc, char *argv[])
{
  // TODO: Handle command line arguments
  int k = atoi(argv[1]);
  Dataset *training_data = load_dataset(argv[2]);
  Dataset *testing_data = load_dataset(argv[3]);
  int num_procs = atoi(argv[4]);
  if (num_procs <= 0)
  {
    fprintf(stderr, "Given 0 or less num_procs\n");
    return 1;
  }
  if (training_data == NULL || testing_data == NULL)
  {
    fprintf(stderr, "Could not load data\n");
    return 1;
  }
  // TODO: Spawn `num_procs` children
  int child_to_parent[num_procs][2]; // reads input from child
  int parent_to_child[num_procs][2]; // writes input to child
  // TODO: Send information to children
  double test_set_size = testing_data->num_items;
  int N = ceil(test_set_size / num_procs);
  int str_index = 0;
  for (int i = 0; i < num_procs; i++)
  {
    if ((test_set_size - N) < 0)
    {
      N = test_set_size;
    }
    if (pipe(child_to_parent[i]) == -1)
    {
      perror("pipe\n");
      exit(1);
    }
    if (pipe(parent_to_child[i]) == -1)
    {
      perror("pipe\n");
      exit(1);
    }
    int result = fork();
    if (result < 0)
    {
      perror("fork\n");
      exit(1);
    }
    else if (result == 0) //child process
    {
      if (close(child_to_parent[i][0]) == -1)
      {
        perror("close read from child end from inside child\n");
        exit(1);
      }
      if (close(parent_to_child[i][1]) == -1)
      {
        perror("close write to child end from inside child\n");
        exit(1);
      }
      for (int child_no = 0; child_no < i; child_no++)
      {
        if (close(child_to_parent[child_no][0]) == -1)
        {
          perror("close read from child ends of previously forked children\n");
          exit(1);
        }
        if (close(parent_to_child[child_no][1]) == -1)
        {
          perror("close write to child ends of previously forked children\n");
          exit(1);
        }
      }
      child_handler(training_data, testing_data, k, parent_to_child[i][0], child_to_parent[i][1]);
      for (int child_no = 0; child_no < i; child_no++)
      {
        if (close(child_to_parent[child_no][1]) == -1)
        {
          perror("close read from child ends of previously forked children\n");
          exit(1);
        }
        if (close(parent_to_child[child_no][0]) == -1)
        {
          perror("close write to child ends of previously forked children\n");
          exit(1);
        }
      }
      if (close(child_to_parent[i][1]) == -1)
      {
        perror("close pipe after writing\n");
        exit(1);
      }
      if (close(parent_to_child[i][0]) == -1)
      {
        perror("close pipe after writing\n");
        exit(1);
      }
      free_dataset(training_data);
      free_dataset(testing_data);
      str_index = str_index + N;
      test_set_size = test_set_size - N;
      exit(0); // Don’t fork children on next loop iteration
    }
  }
  test_set_size = testing_data->num_items;
  N = ceil(test_set_size / num_procs);
  str_index = 0;
  for (int i = 0; i < num_procs; i++) //parent process
  {
    if ((test_set_size - N) < 0)
    {
      N = test_set_size;
    }

    if (close(parent_to_child[i][0]) == -1)
    {
      perror("close read from parent end from inside parent\n");
      exit(1);
    }
    // Now write to child -
    if (write(parent_to_child[i][1], &str_index, sizeof(int)) != sizeof(int))
    {
      perror("write str_index to child from parent\n");
      exit(1);
    }
    if (write(parent_to_child[i][1], &N, sizeof(int)) != sizeof(int))
    {
      perror("write N to child from parent\n");
      exit(1);
    }
    // Close the pipe since we are done with it.
    if (close(parent_to_child[i][1]) == -1)
    {
      perror("close write to child after writing\n");
      exit(1);
    }

    str_index = str_index + N;
    test_set_size = test_set_size - N;
  }
  //parent waits for all children
  for (int i = 0; i < num_procs; i++)
  {
    pid_t pid;
    int status;
    if ((pid = wait(&status)) == -1)
    {
      perror("wait");
      exit(1);
    }
  }
  // TODO: Compute the total number of correct predictions from returned values
  int total_correct = 0;
  // Read one integer from each child, print it and add to sum
  for (int i = 0; i < num_procs; i++)
  {
    int contribution = 0;
    if (close(child_to_parent[i][1]) == -1)
    {
      perror("close writing to parent end of pipe in parent\n");
      exit(1);
    }
    if (read(child_to_parent[i][0], &contribution, sizeof(int)) == -1)
    {
      perror("reading from a pipe from a child\n");
      exit(1);
    }
    total_correct += contribution;
    if (close(child_to_parent[i][0]) == -1)
    {
      perror("close reading from child end of pipe in parent\n");
      exit(1);
    }
  }
  free_dataset(training_data);
  free_dataset(testing_data);
  // Print out answer
  printf("%d\n", total_correct);
  return 0;
}
