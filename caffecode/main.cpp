/*
 * main.cpp
 *
 *  Created on: Apr 24, 2017
 *      Author: savas
 */

//#define CPU_ONLY

#include "cv.h"
#include "highgui.h"

#include "vector"
#include <unistd.h>
#include "fstream"
#include "omp.h"

#include <glob.h>

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "boost/scoped_ptr.hpp"

#include "EError.h"

#include "UBinaryInput.h"
#include "UBinaryOutput.h"

#include "CVideo.h"
#include "UParser.h"
#include "UConfig.h"

#include "cnpy.h"

#include "CSVM.h"

DEFINE_string(backend, "lmdb", "The backend {leveldb, lmdb} containing the images");

using namespace caffe;
using boost::scoped_ptr;

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32,
		dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

class CaffeFace
{
public:

	CaffeFace()
	{}

	void set()
	{
		Caffe::set_mode(Caffe::CPU);
	}

	void set(int gpuid)
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpuid);
	}

	void init(string strprototxt, string strmodel)
	{
		net_.reset(new Net<float>(strprototxt, TEST));
		net_->CopyTrainedLayersFrom(strmodel);

		Blob<float>* input_layer = net_->input_blobs()[0];
		num_channels_   = input_layer->channels();
		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	}

	void init(string strprototxt, string strmodel, string strmean)
	{
		net_.reset(new Net<float>(strprototxt, TEST));
		net_->CopyTrainedLayersFrom(strmodel);

		Blob<float>* input_layer = net_->input_blobs()[0];
		num_channels_   = input_layer->channels();
		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

		set_mean(strmean);
	}

	vector<float> predict(const cv::Mat& imat)
	{
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		wrap(&input_channels);

		preprocess(imat, input_channels);

		net_->ForwardPrefilled();
		Blob<float>* output_layer = net_->output_blobs()[0];

		int ch = output_layer->channels();
		int ht = output_layer->height();
		int wt = output_layer->width();

		const float* begin = output_layer->cpu_data();
		const float* end   = begin + ch*ht*wt;

		return vector<float>(begin, end);
	}

private:

	void wrap(std::vector<cv::Mat>* input_channels)
	{
		Blob<float>* input_layer = net_->input_blobs()[0];

		int width = input_layer->width();
		int height = input_layer->height();

		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i)
		{
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}

	void preprocess(const cv::Mat& img, std::vector<cv::Mat>& input_channels)
	{
		cv::Mat sample_resized;
		if (img.size() != input_geometry_) cv::resize(img, sample_resized, input_geometry_);
		else 							   sample_resized = img;

        //cv::Mat sample_normalized;
		//cv::subtract(sample_resized, mean_, sample_normalized);

		cv::split(sample_resized, input_channels);

		CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
	        == net_->input_blobs()[0]->cpu_data())
	    	<< "Input channels are not wrapping the input layer of the network.";
	}

	void set_mean(string strmean)
	{
		  BlobProto blob_proto;
		  ReadProtoFromBinaryFileOrDie(strmean.c_str(), &blob_proto);

		  /* Convert from BlobProto to Blob<float> */
		  Blob<float> mean_blob;
		  mean_blob.FromProto(blob_proto);
		  CHECK_EQ(mean_blob.channels(), num_channels_)
		    << "Number of channels of mean file doesn't match input layer.";

		  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
		  std::vector<cv::Mat> channels;
		  float* data = mean_blob.mutable_cpu_data();
		  for (int i = 0; i < num_channels_; ++i) {
		    /* Extract an individual channel. */
		    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		    channels.push_back(channel);
		    data += mean_blob.height() * mean_blob.width();
		  }

		  /* Merge the separate channels into a single image. */
		  cv::Mat mean;
		  cv::merge(channels, mean);

		  /* Compute the global mean pixel value and create a mean image
		   * filled with this value. */
		  cv::Scalar channel_mean = cv::mean(mean);
		  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
	}

	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};

void cv2dlib(cv::Mat& icv, dlib::matrix<dlib::rgb_pixel>& ocv)
{
    ocv.set_size(icv.rows, icv.cols);

    dlib::matrix<dlib::rgb_pixel>::iterator it;
    int count = 0;
	for(it = ocv.begin(); it != ocv.end(); it++, count++)
	{
		int h = floor(count/icv.cols);
		int w = count%icv.cols;

		cv::Vec3b v = icv.at<cv::Vec3b>(h,w);

		(*it).red   = v[2];
		(*it).green = v[1];
		(*it).blue  = v[0];
	}
}

void l2norm(vector<float>& feat)
{
	double nr = 0.0;
	int ln = feat.size();
	for(int i=0; i<ln; i++)
		nr += feat[i]*feat[i];

	nr = sqrt(nr);
	if(nr > 1e-13)
	{
		for(int i=0; i<ln; i++)
			feat[i] /= nr;
	}
}

void absolute(vector<float>& feat)
{
	for(int i=0; i<feat.size(); i++)
		feat[i] = abs(feat[i]);
}

const int width = 227;
void dlib2cv(dlib::matrix<dlib::rgb_pixel>& icv, cv::Mat& ocv)
{
    ocv = cv::Mat(width,width,CV_32FC3);
    dlib::matrix<dlib::rgb_pixel>::iterator it;
    int count = 0;
	for(it = icv.begin(); it != icv.end(); it++, count++)
	{
		double r = double((*it).red)/256.0;
		double g = double((*it).green)/256.0;
		double b = double((*it).blue)/256.0;

		int h = floor(count/width);
		int w = count%width;

		cv::Vec3f vec = {b,g,r}; //todo
		ocv.at<cv::Vec3f>(h,w) = vec;
	}
}

void detect_face(string file, net_type& net, const dlib::shape_predictor& sp,
		         vector<cv::Mat>& vface)
{
	dlib::matrix<dlib::rgb_pixel> img;
	cv::Mat cvimg  =  cv::imread(file, -1);
	cv2dlib(cvimg, img);

    for(auto face : net(img))
    {
        auto shape = sp(img, face);
        dlib::matrix<dlib::rgb_pixel> facesp;
        extract_image_chip(img, dlib::get_face_chip_details(shape,width,0.25), facesp);

        cv::Mat cvfacesp;
        dlib2cv(facesp, cvfacesp);
        vface.push_back(cvfacesp);
    }
}

void detect_face(cv::Mat cvimg, net_type& net, const dlib::shape_predictor& sp,
		         vector<cv::Mat>& vface)
{
	dlib::matrix<dlib::rgb_pixel> img;
	cv2dlib(cvimg, img);

    for(auto face : net(img))
    {
        auto shape = sp(img, face);
        dlib::matrix<dlib::rgb_pixel> facesp;
        extract_image_chip(img, dlib::get_face_chip_details(shape,width,0.25), facesp);

        cv::Mat cvfacesp;
        dlib2cv(facesp, cvfacesp);
        vface.push_back(cvfacesp);
    }
}

void extract_feat(cv::Mat& face, CaffeFace& cnn, vector<float>& vfeat)
{
	vfeat = cnn.predict(face);
}

const string sfacedetector = "mmod_human_face_detector.dat";
const string sshapedetector = "shape_predictor_68_face_landmarks.dat";

const string scaffeprotoemot = "Submission_3_deploy.prototxt";
const string scaffemodelemot = "Submission_3.caffemodel";
const string scaffemeanemot  = "Submission_3_mean.binaryproto";

int str2int(string sval)
{
	int ival;
	istringstream ss(sval);
	ss >> ival;

	return ival;
}

cv::Mat convert_cimage_to_iplimage(const CFrame& cImg, const CFrame::EFormat& format = CImage::RGB)
{
	if(cImg.Format() != CImage::RGB) throw EError("[] Input image format must be RGB.");

	IplImage cvImage;

	{
		cvInitImageHeader
		(
			&cvImage,
			cvSize(cImg.Width(), cImg.Height()),
			8,
			4,
			IPL_ORIGIN_TL,
			4
		) ;
		cvImage.imageData = reinterpret_cast<char*>(const_cast<CImage::TComponent*>(cImg.Data())) ;
	}

	IplImage* pCvImage;

	switch (format)
	{
		case CImage::RGB:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 3);
			cvCvtColor(&cvImage, pCvImage,  CV_RGBA2BGR);
			break;
		}
		case CImage::RGBA:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 4);
			cvCvtColor(&cvImage, pCvImage,  CV_RGB2BGRA);
			break;
		}
		case CImage::Grayscale:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 1);
			cvCvtColor(&cvImage, pCvImage,  CV_RGB2GRAY);
			break;
		}
		break;
		case CImage::YUV:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 3);
			cvCvtColor(&cvImage, pCvImage,  CV_RGB2YUV);
			break;
		}
		break;
		case CImage::XYZ:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 3);
			cvCvtColor(&cvImage, pCvImage,  CV_RGB2XYZ);
			break;
		}
		break;
		case CImage::HSV:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 3);
			cvCvtColor(&cvImage, pCvImage,  CV_RGB2HSV);
			break;
		}
		break;
		case CImage::LAB:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 3);
			cvCvtColor(&cvImage, pCvImage,  CV_RGB2Lab);
			break;
		}
		case CImage::LUV:
		{
			pCvImage = cvCreateImage(cvGetSize(&cvImage), IPL_DEPTH_8U, 3);
			cvCvtColor(&cvImage, pCvImage,  CV_RGB2Luv);
			break;
		}
		break;
		default:
		{
			throw EError("[ConvertCImage2Iplimage] Undefined color type");
		}
	}

	return cv::Mat(pCvImage);
}

double total_sum(const vector<float>& vsum, int nt)
{
	double dsum = 0.0;
	for(int i=0; i<nt; i++)
		dsum += vsum[i];

	return dsum / double(nt);
}

vector<float> element_mean(const vector<vector<float> >& velem, int nt)
{
	int ncount = velem.size();
	vector<float> vsum(nt, 0.0);
	for(int i=0; i<ncount; i++)
	{
		vsum += velem[i];
	}

	return vsum / float(ncount) ;
}

vector<float> element_var(const vector<vector<float> >& velem, const vector<float>& vmean, int nt)
{
	int ncount = velem.size();
	vector<float> vvar(nt, 0.0);
	for(int i=0; i<ncount; i++)
	{
		vvar += (velem[i]-vmean)*(velem[i]-vmean);
	}

	for(int i=0; i<nt; i++)
		vvar[i] = sqrt(vvar[i]);

	return vvar / float(ncount) ;
}

vector<float> element_max(const vector<vector<float> >& velem, int nt)
{
	int ncount = velem.size();
	vector<float> vsum(nt, 0.0);
	for(int i=0; i<ncount; i++)
	{
		for(int j=0; j<velem[i].size(); j++)
			vsum[j] = max(vsum[j], velem[i][j]);
	}

	return vsum;
}

float sign(const float& v)
{
	return (v >= 0) ? 1 : -1;
}

float power(const float& v)
{
	return pow(abs(v), 0.5);
}

void power_norm(vector<float>& vcode)
{
	for(int i=0; i<int(vcode.size()); i++)
		vcode[i] = sign(vcode[i])*power(vcode[i]);
}


vector<float> dot_prod(const vector<float>& v1, const vector<float>& v2, int nt)
{
	vector<float> vp(nt*(nt+1)/2, 0.0);
	int cntr = 0;
	for(int i1=0; i1<nt; i1++)
		for(int i2=i1; i2<nt; i2++, cntr++)
			vp[cntr] = v1[i1]*v2[i2];

	return vp;
}

vector<float> covrepr(const vector<vector<float> >& vfeat1, const vector<vector<float> >& vfeat2, int nt)
{
	int ncount = vfeat1.size();
	vector<float> vm1 = element_mean(vfeat1, nt);
	vector<float> vm2 = element_mean(vfeat2, nt);
	vector<float> vc(nt*(nt+1)/2, 0.0);
	for(int i=0; i<ncount; i++)
	{
		vector<float> vdiff1 = vfeat1[i]-vm1;
		vector<float> vdiff2 = vfeat2[i]-vm2;
		vc += dot_prod(vdiff1, vdiff2, nt);
	}

	vc /= float(ncount);

	return vc;
}

vector<float> covrepr(const vector<vector<float> >& vfeat, int nt)
{
	int ncount = vfeat.size();
	vector<float> vm = element_mean(vfeat, nt);
	vector<float> vc(nt*(nt+1)/2, 0.0);
	for(int i=0; i<ncount; i+=1)
	{
		vector<float> vdiff = vfeat[i]-vm;
		//l2norm(vdiff);
		vc += dot_prod(vdiff, vdiff, nt);
	}

	vc /= float(ncount);

	return vc;
}

vector<float> maxrepr(const vector<vector<float> >& vfeat, int nt)
{
	return element_max(vfeat, nt);
}

void read_npydata(const string& pthfile, vector<vector<float> >& vdatai)
{
	vdatai.clear();
	cnpy::NpyArray arr = cnpy::npy_load(pthfile);
	float* data = reinterpret_cast<float*>(arr.data);

	int nelem = arr.shape[0];
	int ndim  = arr.shape[1];

	for(int i=0; i<nelem; i++)
	{
		vector<float> vdata(data+i*ndim, data+(i+1)*ndim);
		vdatai.push_back(vdata);
	}
}

vector<float> facetoupper(const vector<float>& vcode, int nc, int nw, int nh)
{
	vector<float> vface(nc*nw*nh/2, 0.0);

	int cntr = 0;
	for(int h=0; h<nh/2; h++)
		for(int w=0; w<nw; w++)
				for(int c=0; c<nc; c+=1, cntr++)
				{
					float val = vcode[(c)*nh*nw + h*nw + w];
					vface[cntr] = val;
				}

	return vface;
}

vector<float> face3dcnn(const vector<float>& vcode, int nc, int nw, int nh)
{
	vector<float> vface(nc*nw*nh, 0.0);

	int cntr = 0;
	for(int h=0; h<nh; h++)
		for(int w=0; w<nw; w++)
			for(int c=0; c<nc; c+=1, cntr++)
			{
				float val = vcode[(c)*nh*nw + h*nw + w];
				vface[cntr] = val;
			}

	return vface;
}

vector<float> facetolower(const vector<float>& vcode, int nc, int nw, int nh)
{
	vector<float> vface(nc*nw*nh/2, 0.0);

	int cntr = 0;
	for(int h=nh/2; h<nh; h++)
		for(int w=0; w<nw; w++)
				for(int c=0; c<nc; c+=1, cntr++)
				{
					float val = vcode[(c)*nh*nw + h*nw + w];
					vface[cntr] = val;
				}

	return vface;
}

void compute_emotion_features(UConfig& config)
{
	int thread = 8;
	omp_set_num_threads(thread);

	const int emperiod = 200;
	const int emstep   = 20;

	vector<net_type> vfacenet(thread);
    for(int t=0; t<thread; t++)
    	dlib::deserialize(sfacedetector.c_str()) >> vfacenet[t];

	vector<dlib::shape_predictor> vfacesp(thread);
	for(int t=0; t<thread; t++)
		dlib::deserialize(sshapedetector.c_str()) >> vfacesp[t];

    vector<CaffeFace> vfeatcnn(thread);
    for(int t=0; t<thread; t++)
    {
    	vfeatcnn[t].set(1);
    	vfeatcnn[t].init(scaffeprotoemot, scaffemodelemot, scaffemeanemot);
    }

    string type;
    vector<UPath> vlist;
    vector<int> vlabel;

    UParser parse(UPath(config.Read<string>("file")));
    type = parse.Parse<string>();
    while(not parse.IsEnded())
    {
    	vlist.push_back(parse.Parse<string>());
    	vlabel.push_back(parse.Parse<int>());
    }

    int nvideo = int(vlist.size());

	#pragma omp parallel
	{
		#pragma omp for
		for(int vid = 0; vid<nvideo; vid++)
		{
			int tid = omp_get_thread_num();

			UPath pthfile(type, string(vlist[vid].BaseName().RawName()) + ".dat");
			//UPath pthfile(type, string(UParser(vid))+"."+string(UParser(vlabel[vid])) + ".dat");

			if( pthfile.Exists() )
			{
				cout<<"Skipping... "<<vid<<endl;
				continue;
			}

			sleep(tid);

			CVideo video;
			video.Open( vlist[vid] );

			int nframe = video.FrameCount();
			int nstart = floor((nframe - emperiod)/2);

			cout<<vid<<" "<<tid<<endl;

			vector<vector<float> > vtensorfeat;
			for(int i=nstart; i<nstart+emperiod+5; i+=1)
			{
				CFrame frame = video.GetFrame(i);
				frame.ConvertTo(CImage::RGB);
				cv::Mat cvframe = convert_cimage_to_iplimage(frame);

				vector<cv::Mat> vfaces;
				detect_face(cvframe, vfacenet[tid], vfacesp[tid], vfaces);

				int nfaces = int(vfaces.size());
				if(nfaces != 1) std::cout<<"warning: "<<vid<<" #faces:"<<nfaces<<std::endl;
				if(nfaces == 0) continue;

				cv::Mat grayMat, bgrMat;
				cv::cvtColor(vfaces[0], grayMat, CV_BGR2GRAY);
				cv::cvtColor(grayMat, bgrMat, CV_GRAY2BGR);

				vector<float> vfeat;
				extract_feat(bgrMat, vfeatcnn[tid], vfeat);

				vtensorfeat.push_back(vfeat);
			}

			UBinaryOutput output(pthfile);
			output.Write(vtensorfeat);
			output.Close();
		}
	}
}

void compact_pca(UConfig& config)
{
	UPath pthdirec = config.Read<string>("direc");
	vector<UPath> vfilelist = UPath(pthdirec, "*.dat").Glob();

	int nchannel = 256;
	int interval = 20;
	int npart = nchannel*6*6;

	int nfile = int(vfilelist.size());
	for(int v=0; v<nfile; v++)
	{
		cout<<vfilelist[v].BaseName()<<endl;

		int label = UParser(string(vfilelist[v].BaseName().RawName().Extension())).Parse<int>();
		vector<vector<float> > vtensorfeat;

		UBinaryInput in(vfilelist[v]);
		in.Read(vtensorfeat);
		in.Close();

		int nduration = int(vtensorfeat.size());
		vector<vector<float> > vf;
		for(int i=0; i<nduration; i+=1)
			vf.push_back(vtensorfeat[i]);

		vector<float> ovf = element_mean(vf, npart);


	}
}

void compact_training_npy(UConfig& config)
{
	UPath pthdirec = config.Read<string>("direc");
	vector<UPath> vfilelist = UPath(pthdirec, "*.dat").Glob();

	vector<vector<float> > vlstmlower;
	vector<int> vlabels;

	int nchannel = 256;
	int interval = 10;
	int npart = nchannel*6*6;

	int ngroup = 15;
	int nsub   = 1;
	int nstep  = nsub*ngroup;

	int nfile = int(vfilelist.size());
	for(int v=0; v<nfile; v++)
	{
		cout<<vfilelist[v].BaseName()<<endl;

		int label = UParser(string(vfilelist[v].BaseName().RawName().Extension())).Parse<int>();
		vector<vector<float> > vtensorfeat;

		UBinaryInput in(vfilelist[v]);
		in.Read(vtensorfeat);
		in.Close();

		int nduration = int(vtensorfeat.size());
		vector<vector<float> > vf;
		for(int i=0; i<nduration; i+=1)
			vf.push_back(vtensorfeat[i]);

		vector<float> ovf = element_mean(vf, npart);

		for(int i=20; i<nduration-20; i+=interval)
		{
			if(i+nstep >= nduration) continue;

			vector<float> vfeatlower;

			for(int p=0; p<ngroup; p++)
			{
				vector<float> fd = (vtensorfeat[i+p] - ovf);
				vector<float> vml = face3dcnn(fd, nchannel, 6, 6); //element_mean(vftl, npart/4); //l2norm(vml);
				vfeatlower.insert(vfeatlower.end(), vml.begin(), vml.end());
			}

			vlstmlower.push_back(vfeatlower);
			vlabels.push_back(label);
		}
	}
	cout<<"Feat : "<<vlstmlower.size()<<endl;
	cout<<"Label: "<<vlabels.size()<<endl;

	int Nx = int(vlstmlower.size());
    int Nf = npart*ngroup;

    const unsigned int shape_data[] = {Nx, Nf};

    float* datal = new float[Nx*Nf];
    for(int x=0, i=0; x<Nx; x++)
    	for(int f=0; f<Nf; f++, i++)
    		datal[i] = vlstmlower[x][f];

    cnpy::npy_save("data.npy", datal, shape_data, 2, "w");

    int* label = new int[Nx];
    for(int i=0; i<Nx; i++)
    	label[i] = vlabels[i];

    const unsigned int shape_label[] = {Nx};
    cnpy::npy_save("label.npy", label, shape_label, 1, "w");
}

void compact_test_val_npy(UConfig& config)
{
	UPath pthdirec = config.Read<string>("direc");
	string pthout  = config.Read<string>("output");

	vector<UPath> vfiles = UPath(pthdirec, "*.dat").Glob();

	for(int f=0; f<vfiles.size(); f++)
	{
		vector<vector<float> > lstmlower;

		int nchannel = 256;
		int interval = 10;
		int npart = nchannel*6*6;

		int ngroup = 15;
		int nsub   = 1;
		int nstep  = nsub*ngroup;

		vector<vector<float> > vtensorfeat;

		UBinaryInput in(vfiles[f]);
		in.Read(vtensorfeat);
		in.Close();

		int nduration = int(vtensorfeat.size());
		vector<vector<float> > vf;
		for(int i=0; i<nduration; i+=1)
			vf.push_back(vtensorfeat[i]);

		vector<float> ovf = element_mean(vf, npart);

		for(int i=10; i<nduration-10; i+=interval)
		{
			if(i+nstep >= nduration) continue;

			vector<float> vfeatlower;

			for(int p=0; p<ngroup; p++)
			{
				vector<float> fd = (vtensorfeat[i+p] - ovf);
				vector<float> vml = face3dcnn(fd, nchannel, 6, 6); //element_mean(vftl, npart/4); //l2norm(vml);
				vfeatlower.insert(vfeatlower.end(), vml.begin(), vml.end());
			}

			lstmlower.push_back(vfeatlower);
		}

	    int Nx = int(lstmlower.size());
	    int Nf = npart*ngroup;

	    cout<<"size: "<<Nx<<endl;

	    const unsigned int shape_data[] = {Nx, Nf};

	    float* datal = new float[Nx*Nf];
	    for(int x=0, i=0; x<Nx; x++)
	    	for(int f=0; f<Nf; f++, i++)
	    		datal[i] = lstmlower[x][f];

	    cnpy::npy_save(string(UPath(pthout, vfiles[f].BaseName().RawName() + ".npy")), datal, shape_data, 2, "w");
	}
}

int main(int argc, char** argv)
{
	UConfig config(argc, argv);
//	compute_emotion_features(config);
//	compact_training_npy(config);
//	compact_test_val_npy(config);
}

