#pragma once

#include "UParser.h"
#include "UPath.h"

#include "EError.h"

#include "extra/libsvm.h"

#include "stdlib.h"
#include "memory.h"

class CSVM
{
public:

	typedef vector<float> t_sample;
	typedef int t_label;
	typedef vector<double> t_confidence;

	CSVM(int svmtype=ONE_CLASS, int kerneltype=RBF, int degree=1, double gamma=0,
		 double coef0=0.0, double nu=0.5, double cachesize=1000, double C=0.01,
		 double eps=1e-3, double p=0.1, int shrinking=1, int probability=0,
		 int nrweight=0, int* weigthlabel=NULL, double* weigth=NULL);

	virtual ~CSVM();

	void train(const vector<t_sample>& v_samples, const vector<t_label>& v_labels);
	void cross_validation(const vector<t_sample>& v_samples, const vector<t_label>& v_labels,
			   	   	   	  const int& nfold, vector<t_label>& v_labels_up);

	t_label predict(const t_sample& sample);
	vector<t_label> predict(const vector<t_sample>& v_sample);

	t_confidence  predict_probability(const t_sample& sample);
	vector<t_confidence> predict_probability(const vector<t_sample>& v_samples);

	void read(const UPath& pth_input);
	void write(const UPath& pth_output);

private:

	static const float m_epsilon;

	/// Computes l2 norm for a given vector.
	template<typename T>
	double l2_norm(const vector<T>& vec)
	{
		return sqrt( UStatistics(vec).SquareSum() +  m_epsilon*m_epsilon);
	}

	/// Computes l1 norm for a given vector.
	template<typename T>
	double l1_norm(const vector<T>& vec)
	{
		return UStatistics(vec).Sum() + m_epsilon;
	}

	bool m_b_trained;
	svm_model* m_model;
	svm_node* m_xspace;

	svm_parameter m_param;
};
