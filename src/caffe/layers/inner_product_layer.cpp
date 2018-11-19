#include <vector>
//save file for debugging
#include <fstream>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // Dtype* weight = this->blobs_[0]->mutable_cpu_data();

//  std::cout <<"output some to test weigts "<< weight[0]<<" "<< weight[100]<<" "<<weight[200]<<"\n";

  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }

//  std::cout <<"output some to top_data "<< top_data[0]<<" "<< top_data[100]<<" "<<top_data[200]<<"\n";
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  std::cout<<"\n backward cpu called param_propagate_down_[0] "<< this->param_propagate_down_[0]
      << " param_propagate_down_[1] "<<this->param_propagate_down_[1] <<"\n";
      std::cout<<"\nblobs size "<<this->blobs_.size()<<"\n";
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

//my implementation for backpropagate of DELTA Error
template <typename Dtype>
std::pair<Dtype, Dtype*> InnerProductLayer<Dtype>::BackwardError_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom, const Dtype delta_top, const Dtype W_portion ) {


  const Dtype* weights  = this->blobs_[0]->cpu_data();
  //second_delta = np.max(w3.T,axis=0) size = A2 ; input of W3

  Dtype* weights_max = new Dtype[this->blobs_[0]->shape(1)](); //initialize with zero

#ifdef DEBUG_CODE
  std::cout<<"\n delta_top "<<delta_top<<"\n";
  std::cout<<"\nblobs size: "<<this->blobs_.size()<<"\n";
  for (int i =0;i<this->blobs_.size() ;i++)
      std::cout<<"\nblob : "<<i<<" shape "<<this->blobs_[i]->shape_string()<<"\n";
  std::cout <<"allocating weights_max array size: "<<this->blobs_[0]->shape(1)<<"\n";
  std::cout <<"output some to test "<< weights_max[0] << weights_max[1]<<weights_max[2]<<"\n";
#endif

  Dtype* sum_squared_weights = new Dtype[this->blobs_[0]->shape(0)]();
  Dtype* sum_abs_weights = new Dtype[this->blobs_[0]->shape(0)]();
  for (int i =0; i< this->blobs_[0]->shape(0) ; i++) //10
  {
    for (int j =0 ; j <this->blobs_[0]->shape(1); j++) // 300
    {
      if (weights[i*this->blobs_[0]->shape(1) + j] > weights_max[j])
        weights_max[j] = weights[i*this->blobs_[0]->shape(1) + j];

      sum_abs_weights[i] += fabs(weights[i*this->blobs_[0]->shape(1) + j]);
      sum_squared_weights[i] += weights[i*this->blobs_[0]->shape(1) + j]*weights[i*this->blobs_[0]->shape(1) + j];
    }
  //  std::cout<<"test first col "<< weights[i*this->blobs_[0]->shape(1)]<<"\n";
  }

double  max_sum_squared_weights = 0;
  for (int i =0; i< this->blobs_[0]->shape(0) ; i++) {
    sum_squared_weights[i] = sqrt(sum_squared_weights[i]);
    if (sum_squared_weights[i] > max_sum_squared_weights)
      max_sum_squared_weights = sum_squared_weights[i];
  //  std::cout<<"sum_squared_weights "<<sum_squared_weights[i]<<"\n";
  }

#ifdef DEBUG_CODE
  std::cout <<"output some to test "<< weights_max[0]<<" "<< weights_max[1]<<" "<<weights_max[2]<<"\n";
 //calculate count_nonzero of bottom blobs
  std::cout<<"top size "<<top.size()<<"\n";
  std::cout<<"top shape "<<top[0]->shape_string()<<"\n";
  std::cout<<"bottom size "<<bottom.size()<<"\n";
  std::cout<<"bottom shape "<<bottom[0]->shape_string()<<"\n";
//bottom shape 10000 300 count_nonzero2.shape (10000,)
  std::cout<<"allocate count_nonzero with size "<<bottom[0]->shape(0)<<"\n";
  std::cout<<"allocate avg_bottom with size "<<this->blobs_[0]->shape(1)<<"\n";
#endif

  const Dtype* bottom_vals = bottom[0]->cpu_data();
  Dtype* count_nonzero = new Dtype[bottom[0]->shape(0)](); //initialize with zero


  Dtype* avg_bottom = new Dtype[this->blobs_[0]->shape(1)](); //initialize with zero
  Dtype* count_nonzero_axis1 = new Dtype[this->blobs_[0]->shape(1)](); //initialize with zero

  Dtype* mean_bottom = new Dtype[this->blobs_[0]->shape(1)](); //initialize with zero
  Dtype* std_bottom = new Dtype[this->blobs_[0]->shape(1)](); //initialize with zero

  std::cout << "mean bottom shape "<<this->blobs_[0]->shape(1)<<"\n";
  std::cout << "sample size bottom shape "<<this->blobs_[0]->shape(0)<<"\n";
  for (int i = 0; i<bottom[0]->shape(0) ;i++ )
    for(int j =0; j<this->blobs_[0]->shape(1) ;j++ ){
          mean_bottom[j] += bottom_vals[i*this->blobs_[0]->shape(1) + j];
          if(bottom_vals[i*this->blobs_[0]->shape(1) + j] != 0){
            count_nonzero[i] = count_nonzero[i] + 1;
            count_nonzero_axis1[j] = count_nonzero_axis1[j]+1;
          }
          avg_bottom[j] += fabs(bottom_vals[i*this->blobs_[0]->shape(1) + j]); //fix absolute value, cant assume bottom > 0
        }
  //get average value from sum
  for(int j =0; j<this->blobs_[0]->shape(1) ;j++ ){
        avg_bottom[j]  = avg_bottom[j]/bottom[0]->shape(0);
        mean_bottom[j] = mean_bottom[j]/bottom[0]->shape(0);
        //std::cout<<mean_bottom[j]<<"\n";
  }

  for (int i = 0; i<bottom[0]->shape(0) ;i++ )
    for(int j =0; j<this->blobs_[0]->shape(1) ;j++ ){
          std_bottom[j] += (bottom_vals[i*this->blobs_[0]->shape(1) + j] - mean_bottom[j])*(bottom_vals[i*this->blobs_[0]->shape(1) + j] - mean_bottom[j]);
        }
//det standard deviation value

for(int j =0; j<this->blobs_[0]->shape(1) ;j++ ){
      std_bottom[j] = sqrt(std_bottom[j]/(bottom[0]->shape(0)-1));
  //    std::cout<<std_bottom[j]<<"\n";
}

/*
  std::ofstream myfile;
  myfile.open ("/home/minh/github/caffe/debugging/data.txt");
  for(int j =0; j <300; j ++ )
    myfile << count_nonzero_axis1[j]<<",";

  myfile.close();
*/

//  std::cout <<"output some to test avg_bottom "<< avg_bottom[0]<<" "<< avg_bottom[100]<<" "<<avg_bottom[200]<<"\n";
//  std::cout <<"output some to test count_nonzero "<< count_nonzero[0]<<" "<< count_nonzero[100]<<" "<<count_nonzero[200]<<"\n";
//  std::cout <<"output some to test count_nonzero_axis1 "<< count_nonzero_axis1[0]<<" "<< count_nonzero_axis1[100]<<" "<<count_nonzero_axis1[200]<<"\n";
std::cout <<"output some to test sum_squared_weights "<< sum_squared_weights[0]<<" "<< sum_squared_weights[2]<<" "<<sum_squared_weights[5]<<"\n";
std::cout <<"output some to test sum_abs_weights "<< sum_abs_weights[0]<<" "<< sum_abs_weights[2]<<" "<<sum_abs_weights[5]<<"\n";

std::cout<<"max_sum_squared_weights " << max_sum_squared_weights <<"\n";
  //take SQRT of count count_nonzero

  for (int i =0; i<bottom[0]->shape(0) ; i++)
    count_nonzero[i] = sqrt(count_nonzero[i]);

  double sum_avg_bottom = 0;
  for (int i =0; i<this->blobs_[0]->shape(1) ; i++){
  //  std::cout<<count_nonzero_axis1[i]<<"\n";
      //
      avg_bottom[i] = avg_bottom[i]*bottom[0]->shape(0)/count_nonzero_axis1[i] ;
      avg_bottom[i] = delta_top*sqrt(1)/sqrt(this->blobs_[0]->shape(1))/avg_bottom[i];

      sum_avg_bottom += avg_bottom[i]/(delta_top*sqrt(W_portion)); // the coefficient multiplying with delta_top and weight_portion

    }
    std::cout <<"output some to test avg_bottom "<< avg_bottom[0]<<" "<< avg_bottom[100]<<" "<<avg_bottom[200]<<"\n";

    std::cout<<" delta top "<< delta_top << " W_portion "<<W_portion<<"\n";
  std::cout<<" inner_product_layer avg_weight_delta coeff " <<sum_avg_bottom/this->blobs_[0]->shape(1) <<"\n";

#ifdef DEBUG_CODE
  std::cout<<"delta_top "<<delta_top<<"\n";
  std::cout <<"output some to test avg_bottom "<< avg_bottom[0]<<" "<< avg_bottom[100]<<" "<<avg_bottom[200]<<"\n";
#endif

  Dtype* delta_W= new Dtype[this->blobs_[0]->shape(0)*this->blobs_[0]->shape(1)]();//size = blobs_[0];


 //old version
  for (int i =0; i< this->blobs_[0]->shape(0) ; i++) //10
    for (int j =0 ; j <this->blobs_[0]->shape(1); j++) // 300
       delta_W[i*this->blobs_[0]->shape(1) + j] = avg_bottom[j] ;

/*
  for (int i =0; i< this->blobs_[0]->shape(0) ; i++) //10
    for (int j =0 ; j <this->blobs_[0]->shape(1); j++) // 300
    {
       delta_W[i*this->blobs_[0]->shape(1) + j] = 1.07*2.3/(sqrt(300)*std_bottom[j]) ;
       if(i ==0)
        std::cout<<1.07*2.51/(sqrt(300)*std_bottom[j]) <<"\n";
    }
*/

  Dtype average_count_nonzero = caffe_cpu_asum(bottom[0]->shape(0),count_nonzero)/bottom[0]->shape(0);


  Dtype average_weights_max = caffe_cpu_asum(this->blobs_[0]->shape(1),weights_max)/this->blobs_[0]->shape(1);

  Dtype average_sum_abs_weights = caffe_cpu_asum(this->blobs_[0]->shape(1),weights_max)/this->blobs_[0]->shape(1);


    //new_a2 = last_delta/grad_portion/(np.average(np.sqrt(count_nonzero2)))/np.average(second_delta)
  //Dtype delta_bottom = delta_top*(1-W_portion)/average_count_nonzero/average_weights_max;
std::cout<<"old delta "<<delta_top/average_count_nonzero/average_weights_max <<"\n";
//new version
//Wportion = sqrt(1-x).  delta_top = sqrt(W^2 + A^2)
//delta_top^2 = W^2 + A^2
// = a*delta_top^2 + (1-a)*delta_top^2
// sqrt(a)delta_top    |||||  sqrt(1-a)*delta_top
//  Dtype delta_bottom = delta_top*sqrt(1-W_portion)/(caffe_cpu_asum(this->blobs_[0]->shape(0),sum_squared_weights)/this->blobs_[0]->shape(0));//;average_count_nonzero/average_weights_max;

//  Dtype delta_bottom = delta_top/(caffe_cpu_asum(this->blobs_[0]->shape(0),sum_squared_weights)/this->blobs_[0]->shape(0));//;average_count_nonzero/average_weights_max;
  Dtype delta_bottom = delta_top/max_sum_squared_weights;//(caffe_cpu_asum(this->blobs_[0]->shape(0),sum_squared_weights)/this->blobs_[0]->shape(0));//;average_count_nonzero/average_weights_max;
  std::cout<<" avg sum_squared_weights "<<(caffe_cpu_asum(this->blobs_[0]->shape(0),sum_squared_weights)/this->blobs_[0]->shape(0))<<"\n";

  std::cout<<" avg sum_abs_weights "<<(caffe_cpu_asum(this->blobs_[0]->shape(0),sum_abs_weights)/this->blobs_[0]->shape(0))<<"\n";


  std::cout<<" inner_product_layer delta_bottom_coeff " <<delta_bottom/(delta_top*sqrt(1-W_portion)) <<"\n";


//#ifdef DEBUG_CODE
  std::cout <<"average_count_nonzero " << average_count_nonzero <<"\n";
  std::cout <<"average_weights_max " << average_weights_max <<"\n";
//  std::cout <<"delta_bottom " << delta_bottom <<"\n";
//  std::cout<<"returning delta_W with size "<<this->blobs_[0]->shape(0)*this->blobs_[0]->shape(1)<<"\n";
//#endif
//return avg sum_sqaured_weights as delta bottom
  return std::make_pair((caffe_cpu_asum(this->blobs_[0]->shape(0),sum_squared_weights)/this->blobs_[0]->shape(0)) ,delta_W);
      /*
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
  */
}
template <typename Dtype>
std::pair<Dtype, Dtype*> InnerProductLayer<Dtype>::BackwardError_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,const Dtype delta_top, const Dtype W_portion) {

    return   BackwardError_cpu(top, propagate_down, bottom, delta_top,  W_portion);

    }

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
//STUB_GPU_ERROR(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
