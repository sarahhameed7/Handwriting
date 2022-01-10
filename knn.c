#include "knn.h"
int k_value = 0;
/****************************************************************************/
/* For all the remaining functions you may assume all the images are of the */
/*     same size, you do not need to perform checks to ensure this.         */
/****************************************************************************/

/**************************** A1 code ****************************************/

/* Same as A1, you can reuse your code if you want! */

int find_max(int max_entry, int k_entries[k_value][2])
{
  for (int i = 0; i < k_value; i++)
  {
    if (k_entries[i][0] > k_entries[max_entry][0])
    {
      max_entry = i;
    }
  }
  return max_entry;
}
int find_mode(int k, int k_entries[k_value][2])
{
  int mode = 0, max_count = 0;
  for (int i = 0; i < k; ++i)
  {
    int counter = 0;
    for (int j = 0; j < k; ++j)
    {
      if (k_entries[j][1] == k_entries[i][1])
      {
        ++counter;
      }
    }
    if (counter > max_count)
    {
      max_count = counter;
      mode = k_entries[i][1];
    }
  }
  return mode;
}
double distance(Image *a, Image *b)
{
  // TODO: Return correct distance
  int sum = 0;
  for (int i = 0; i < (a->sx) * (a->sy); i++)
  {
    sum += (a->data[i] - b->data[i]) * (a->data[i] - b->data[i]);
  }
  return sqrt(sum);
}

/* Same as A1, you can reuse your code if you want! */
int knn_predict(Dataset *data, Image *input, int K)
{
  // TODO: Replace this with predicted label (0-9)
  k_value = K;
  int k_entries[K][2]; //the two col are for dist and labels
  int max_entry = 0;
  double dist = 0;
  for (int i = 0; i < data->num_items; i++)
  {
    if (i < K)
    {
      dist = distance(input, &(data->images[i]));
      k_entries[i][0] = dist;
      k_entries[i][1] = data->labels[i];
      if (k_entries[i][0] > k_entries[max_entry][0])
      {
        max_entry = i;
      }
    }
    else if (i >= K)
    {
      dist = distance(input, &(data->images[i]));
      if (dist < k_entries[max_entry][0])
      {
        k_entries[max_entry][0] = dist;
        k_entries[max_entry][1] = data->labels[i];
        max_entry = find_max(max_entry, k_entries);
      }
    }
  }

  /*for (int i = 0; i < K; i++)
  {
    printf("dist for k_entries[%d][%d] is :%d \n", i, 0, k_entries[i][0]);
    printf("label for k_entries[%d][%d] is :%d \n", i, 1, k_entries[i][1]);
  }*/
  int mode = find_mode(K, k_entries);
  //printf("the mode for this image is: %d\n",mode);
  return mode;
}

/**************************** A2 code ****************************************/

/* Same as A2, you can reuse your code if you want! */
Dataset *load_dataset(const char *filename)
{
  // TODO: Allocate data, read image data / labels, return

  FILE *file = fopen(filename, "rb");
  if (file == NULL)
  {
    fprintf(stderr, "Could not open file");
    return NULL;
  }
  Dataset *current_data_set = NULL;
  current_data_set = (Dataset *)malloc(sizeof(Dataset));

  int check = fread(&current_data_set->num_items, sizeof(int), 1, file);
  if(check<0)
  {
   fprintf(stderr, "Could not read file");
    return NULL; 
  }
  //printf("numelements = %d\n", current_data_set->num_items);
  current_data_set->labels = (unsigned char *)malloc(current_data_set->num_items * sizeof(unsigned char));
  current_data_set->images = (Image *)malloc(current_data_set->num_items * sizeof(Image));
  for (int i = 0; i < current_data_set->num_items; i++)
  {
    check = fread(&current_data_set->labels[i], sizeof(unsigned char), 1, file);
    if(check<0)
      {
        fprintf(stderr, "Could not read file");
        return NULL; 
      }
    current_data_set->images[i].sx = 28;
    current_data_set->images[i].sy = 28;
    current_data_set->images[i].data = (unsigned char *)malloc(784 * sizeof(unsigned char));
    check = fread(current_data_set->images[i].data, sizeof(unsigned char), 784, file);
    if(check<0)
      {
        fprintf(stderr, "Could not read file");
        return NULL; 
      }
  }
  int error = fclose(file);
  if (error != 0)
  {
    fprintf(stderr, "fclose failed\n");
    return NULL;
  }
  return current_data_set;
}

/* Same as A2, you can reuse your code if you want! */
void free_dataset(Dataset *data)
{
  // TODO: free data
  for (int i = 0; i < data->num_items; i++)
  {
    free(data->images[i].data);
  }
  free(data->images);
  free(data->labels);
  free(data);
  return;
}

/************************** A3 Code below *************************************/

/**
 * NOTE ON AUTOTESTING:
 *    For the purposes of testing your A3 code, the actual KNN stuff doesn't
 *    really matter. We will simply be checking if (i) the number of children
 *    are being spawned correctly, and (ii) if each child is recieving the 
 *    expected parameters / input through the pipe / sending back the correct
 *    result. If your A1 code didn't work, then this is not a problem as long
 *    as your program doesn't crash because of it
 */

/**
 * This function should be called by each child process, and is where the 
 * kNN predictions happen. Along with the training and testing datasets, the
 * function also takes in 
 *    (1) File descriptor for a pipe with input coming from the parent: p_in
 *    (2) File descriptor for a pipe with output going to the parent:  p_out
 * 
 * Once this function is called, the child should do the following:
 *    - Read an integer `start_idx` from the parent (through p_in)
 *    - Read an integer `N` from the parent (through p_in)
 *    - Call `knn_predict()` on testing images `start_idx` to `start_idx+N-1`
 *    - Write an integer representing the number of correct predictions to
 *        the parent (through p_out)
 */
void child_handler(Dataset *training, Dataset *testing, int K,
                   int p_in, int p_out)
{
  // TODO: Compute number of correct predictions from the range of data
  //      provided by the parent, and write it to the parent through `p_out`.
  int str_index = 0;
  int N = 0;
  int total_correct = 0;
  int predict = 0;
  //close wrte to child
  // read from parent
  if (read(p_in, &str_index, sizeof(int)) == -1)
  {
    perror("reading str_index from a parent from inside child handler\n");
    exit(1);
  }
  if (read(p_in, &N, sizeof(int)) == -1)
  {
    perror("reading N from a parent from inside child handler\n");
    exit(1);
  }
  if (N == 0)
  {
  }
  else
  {
    for (int i = str_index; i <= str_index + N - 1; i++)
    {
      predict = knn_predict(training, &testing->images[i], K);
      if (predict == testing->labels[i])
      {
        total_correct++;
      }
    }
  }
  if ((write(p_out, &total_correct, sizeof(int))) == -1)
  {
    perror("writing total correct to parent from inside child handler\n");
    exit(1);
  }
  return;
}