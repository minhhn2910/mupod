#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


//my implementation for backpropagate of DELTA Error
template <typename Dtype>
std::pair<Dtype, Dtype*> ReLULayer<Dtype>::BackwardError_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom, const Dtype delta_top, const Dtype W_portion) {
      Dtype multiply_factor = 1;

      std::cout<<"\n ReLULayer delta_top "<<delta_top<<"\n";
#ifdef DEBUG_CODE
      std::cout<<"\nblobs size: "<<this->blobs_.size()<<"\n";
      for (int i =0;i<this->blobs_.size() ;i++)
          std::cout<<"\nblob : "<<i<<" shape "<<this->blobs_[i]->shape_string()<<"\n";

      std::cout<<"top size "<<top.size()<<"\n";
      std::cout<<"top shape "<<top[0]->shape_string()<<"\n";
      std::cout<<"bottom size "<<bottom.size()<<"\n";
      std::cout<<"bottom shape "<<bottom[0]->shape_string()<<"\n";
#endif
      int n_input = bottom[0]->shape(0);
      int second_dimension = bottom[0]->count()/n_input;

      Dtype* count_nonzero = new Dtype[second_dimension]();

      const Dtype* input_data = bottom[0]->cpu_data();
      for(int i =0;i< n_input; i++){
        for(int j =0; j <second_dimension; j++)
            if(input_data[i*second_dimension+j] >0)
                count_nonzero[j] +=1;
      }


        std::ofstream myfile;
        myfile.open ("/home/minh/github/caffe/debugging/data_relu.txt");
        for(int j =0; j <second_dimension; j ++ )
          myfile << count_nonzero[j]<<",";
        myfile.close();

      Dtype average_count_nonzero = caffe_cpu_asum(second_dimension,count_nonzero)/second_dimension;
      std::cout<<"average count nonzero "<< average_count_nonzero<<" \n";

      multiply_factor = sqrt(float(n_input)/average_count_nonzero);
      std::cout <<"relu multiply factor "<<multiply_factor<<"\n";

      Dtype result[] = {0.0};
      return std::make_pair(delta_top*multiply_factor *2 ,result); //*2 for compatibility with other Layers
      // if a layer has weight, its delta will be divided by 2. while relu doesn't have weight, we keep the api simple, divide by 2 for every layer.
      //multiply by 2 here is to compensate that uniform api
      //return std::make_pair(0,delta_W);
    }




template <typename Dtype>
std::pair<Dtype, Dtype*> ReLULayer<Dtype>::BackwardError_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom,const Dtype delta_top, const Dtype W_portion) {

    return   BackwardError_cpu(top, propagate_down, bottom, delta_top, W_portion);

    }

#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
