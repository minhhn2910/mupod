#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}
/*
//my implementation for backpropagate of DELTA Error, not used
template <typename Dtype>
std::pair<Dtype, Dtype*> ConvolutionLayer<Dtype>::BackwardError_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom, const Dtype delta_top, const Dtype W_portion) {
    //  Dtype grad_portion = 2;

      std::cout<< " W_portion "<< W_portion << "\n";
#ifdef DEBUG_CODE
      std::cout<<"\n ConvolutionLayer delta_top "<<delta_top<<"\n";
      std::cout<<"\nblobs size: "<<this->blobs_.size()<<"\n";
      for (int i =0;i<this->blobs_.size() ;i++)
          std::cout<<"\nblob : "<<i<<" shape "<<this->blobs_[i]->shape_string()<<"\n";
#endif
      const Dtype* weights  = this->blobs_[0]->cpu_data();
      Dtype* delta_W= new Dtype[this->blobs_[0]->count()]();//size = blobs_[0];
      //for(int i =0;i<90 ; i++)
      //  std::cout<<weights[i]<<",";
      std::cout<<"\n";
      Dtype max_sum = 0;
      Dtype sum_of_sum = 0;
      double* sum_squared_weights = new double [this->blobs_[0]->shape(0)]();
      double max_sum_squared_weight = 0;
      double avg_1_per_sum_squared = 0;
      for (int i = 0; i <this->blobs_[0]->shape(0); i++) //output channels
      {
          Dtype sum = 0;
          for(int j = 0; j < this->blobs_[0]->shape(1) * this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3); j++){
            sum += weights[i*this->blobs_[0]->shape(1) * this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3) + j];

            sum_squared_weights[i]+=weights[i*this->blobs_[0]->shape(1) * this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3) + j]*weights[i*this->blobs_[0]->shape(1) * this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3) + j];
          }

          if(sum_squared_weights[i]>max_sum_squared_weight)
            max_sum_squared_weight = sum_squared_weights[i];
          if (fabs(sum)>max_sum)
            max_sum = fabs(sum);

          sum_of_sum += fabs(sum);
          avg_1_per_sum_squared += 1.0/sum_squared_weights[i];
          //std::cout<<"avg_1_per_sum_squared "<< i << " : "<<avg_1_per_sum_squared<<"\n";

          std::cout<<"sum sqared channel "<< i << " : "<<sum_squared_weights[i]<<"\n";

#ifdef DEBUG_CODE
          std::cout<<"sum channel "<< i << " : "<<sum<<"\n";
#endif
      }
      double avg_sum_squared_weights = caffe_cpu_asum(this->blobs_[0]->shape(0),sum_squared_weights)/(this->blobs_[0]->shape(0));
      std::cout<<"avg sum sqared channel " << " : "<<avg_sum_squared_weights<<"\n";
#ifdef DEBUG_CODE
      std::cout<<"result max_sum "<<max_sum << " return "<< delta_top/max_sum <<"\n";
      std::cout<<"bottom shape "<<bottom[0]->shape_string()<<"\n";
#endif
      const Dtype* bottom_vals = bottom[0]->cpu_data();
      //find max input

      Dtype* max_input = new Dtype[bottom[0]->shape(0)*bottom[0]->shape(1)]();
      for (int i =0 ; i<bottom[0]->shape(0)*bottom[0]->shape(1) ; i++ ){
        Dtype local_max = 0;
        for(int j = 0; j < bottom[0]->shape(2)*bottom[0]->shape(3); j++){

            if(fabs(bottom_vals[i*(bottom[0]->shape(2)*bottom[0]->shape(3)) + j]) > local_max)
              local_max = fabs(bottom_vals[i*(bottom[0]->shape(2)*bottom[0]->shape(3)) + j]);

            }
        max_input[i] = local_max;
      };

      Dtype* sum_max_input = new Dtype[bottom[0]->shape(1)]();
      for (int i=0;i<bottom[0]->shape(0);i++ ){
        for(int j =0; j< bottom[0]->shape(1); j++){

          //sum_max_input[j] += max_input[i*bottom[0]->shape(1) + j];

          if(max_input[i*bottom[0]->shape(1) + j]>sum_max_input[j])
            sum_max_input[j] = max_input[i*bottom[0]->shape(1) + j];
        }
      }

      std::ofstream myfile;
      myfile.open ("/home/minh/github/caffe/debugging/max_input.txt");
    //  std::cout<<"\n sum_max_input \n";
      for(int j =0; j <bottom[0]->shape(1); j ++ ){
      //  sum_max_input[j] = sum_max_input[j]/bottom[0]->shape(0); //uncomment when using average

        myfile << sum_max_input[j]<<",";
        //std::cout<<sum_max_input[j]<<", ";
      }
      myfile.close();
      std::cout<<"\n";

      Dtype average_max_input = caffe_cpu_asum(bottom[0]->shape(0),max_input)/(bottom[0]->shape(0));

      avg_1_per_sum_squared = avg_1_per_sum_squared/this->blobs_[0]->shape(0);

      std::cout<<"avg_1_per_sum_squared " <<avg_1_per_sum_squared << "\n";
      std::cout<<"_1_peravg_sum_squared " <<1/avg_sum_squared_weights << "\n";
      double sum_delta_w = 0;
      for (int i =0 ; i<this->blobs_[0]->shape(0); i++){
          for (int j =0 ; j< this->blobs_[0]->shape(1); j++)
              for (int k =0; k<this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3); k++ ){

                  delta_W[(i*this->blobs_[0]->shape(1) + j)*(this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3)) + k]
                    = delta_top*(W_portion)/sum_max_input[j]/(this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3));

                    sum_delta_w += 1.0/(sum_max_input[j]/(this->blobs_[0]->shape(2)* this->blobs_[0]->shape(3)));
                }
            }

      std::cout<<" conv avg_weight_delta coeff " <<caffe_cpu_asum(this->blobs_[0]->count(),delta_W)/(delta_top*(W_portion))/this->blobs_[0]->count() <<"\n";

      std::cout<<" conv delta_bottom_coeff " <<1.0/(sum_of_sum/this->blobs_[0]->shape(0)) <<"\n";

      std::cout<<"delta top "<< delta_top << " result "<< delta_top*avg_1_per_sum_squared;
      return std::make_pair(avg_sum_squared_weights,delta_W);
  //    return std::make_pair(delta_top*(1-W_portion)/(sum_of_sum/this->blobs_[0]->shape(0)),delta_W);
      //return std::make_pair(0,delta_W);
    }


template <typename Dtype>
std::pair<Dtype, Dtype*> ConvolutionLayer<Dtype>::BackwardError_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,const Dtype delta_top, const Dtype W_portion) {

    return   BackwardError_cpu(top, propagate_down, bottom, delta_top, W_portion);

    }
*/
#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
