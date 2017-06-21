/*
 * CSVM.cpp
 *
 *  Created on: Sep 20, 2013
 *      Author: savasozkan
 */

#include "CSVM.h"

using namespace std;

#define Malloc(type, size) (type*)malloc(size*sizeof(type))

const float CSVM::m_epsilon = 1e-6;

CSVM::CSVM(int svmtype, int kerneltype, int degree, double gamma,
		   double coef0, double nu, double cachesize, double C,
		   double eps, double p, int shrinking, int probability,
		   int nrweight, int* weigthlabel, double* weigth)
{
	m_param.svm_type     = svmtype;
	m_param.kernel_type  = kerneltype;
	m_param.degree       = degree;
	m_param.gamma        = gamma;
	m_param.coef0        = coef0;
	m_param.nu           = nu;
	m_param.cache_size   = cachesize;
	m_param.C            = C;
	m_param.eps          = eps;
	m_param.p            = p;
	m_param.shrinking    = shrinking;
	m_param.probability  = probability;
	m_param.nr_weight    = nrweight;
	m_param.weight_label = weigthlabel;
	m_param.weight       = weigth;

	m_model = NULL;
	m_xspace = NULL;

	m_b_trained= false;
}

CSVM::~CSVM()
{
	if(m_model != NULL)
		svm_free_and_destroy_model(&m_model);

	if(m_xspace != NULL)
		free(m_xspace);
}

void CSVM::train(const vector<t_sample>& v_samples, const vector<t_label>& v_labels)
{
	svm_problem prob;

	int size_sample = int(v_samples.size());
	int size_label  = int(v_labels.size());

	if( size_sample != size_label )
		throw error("[CSVM::Train] Index and sample size must be equal.");

	int len = int(v_samples[0].size());

	const char *error_msg = svm_check_parameter(&prob,&m_param);
	if(error_msg)
		throw error("[CSVM::Train] Error occurs as " + string(error_msg));

	prob.l   = size_sample;
	prob.y   = Malloc(double, size_sample);
	prob.x   = Malloc(struct svm_node *, size_sample);
	m_xspace = Malloc(struct svm_node, size_sample*(len+1));

	int j=0;
	for(int i1=0;i1<prob.l;i1++)
	{
		prob.x[i1] = &m_xspace[j];
		prob.y[i1] = v_labels[i1];

		for(int i2=0; i2<len; i2++)
		{
			m_xspace[j].index = i2+1;
			m_xspace[j].value = v_samples[i1][i2];

			++j;
		}

		m_xspace[j++].index = -1;
	}

	m_param.gamma = 1.0/float(len);

	m_model = svm_train(&prob, &m_param);

	free(prob.y);
	free(prob.x);

	m_b_trained = true;
}

void CSVM::cross_validation(const vector<t_sample>& v_samples, const vector<t_label>& v_labels,
						    const int& nfold, vector<t_label>& v_label_up)
{
	if( nfold < 3 )
		throw error("[CSVM::CrossValidation] Fold value must be larger than 2");

	if(m_param.svm_type == EPSILON_SVR || m_param.svm_type == NU_SVR)
		throw error("[] System not adapted for these methods.");

	svm_problem prob;

	int size_sample = int(v_samples.size());
	int size_index  = int(v_labels.size());

	if( size_sample != size_index )
		throw error("[CSVM::Train] Index and sample size must be equal.");

	int len = int(v_samples[0].size());

	const char *error_msg = svm_check_parameter(&prob,&m_param);
	if(error_msg)
		throw error("[CSVM::Train] Error occurs as " + string(error_msg));

	prob.l   = size_sample;
	prob.y   = Malloc(double, size_sample);
	prob.x   = Malloc(struct svm_node *, size_sample);
	m_xspace = Malloc(struct svm_node,   size_sample*(len+1));

	int j=0;
	for(int i1=0;i1<prob.l;i1++)
	{
		prob.x[i1] = &m_xspace[j];
		prob.y[i1] = v_labels[i1];

		for(int i2=0; i2<len; i2++)
		{
			m_xspace[j].index = i2+1;
			m_xspace[j].value = v_samples[i1][i2];

			j+=1;
		}

		m_xspace[j].index = -1;
	}

	if(m_param.gamma == 0 && len > 0)
		m_param.gamma = 1.0/len;

	{
		v_label_up.clear();
		double *target = Malloc(double, prob.l);

		//cout<<"cross validation"<<endl;
		svm_cross_validation(&prob, &m_param, nfold, target);

		for(int i=0; i<prob.l; i++)
			v_label_up.push_back( target[i] );

		free(target);
	}

	free(prob.y);
	free(prob.x);
}

CSVM::t_label CSVM::predict(const t_sample& sample)
{
	if(not m_b_trained)
		throw error("[CSVM::Write] First train the SVM.");

	int len  = int(sample.size());
	m_xspace = (struct svm_node *) realloc(m_xspace, (len+1)*sizeof(struct svm_node));

	for(int i=0; i<len; i++)
	{
		m_xspace[i].index = i+1;
		m_xspace[i].value = sample[i];
	}

	m_xspace[len].index = -1;
	t_label index = svm_predict(m_model, m_xspace);

	return index;
}

vector<CSVM::t_label> CSVM::predict(const vector<t_sample>& v_samples)
{
	vector<CSVM::t_label> vIndex;
	for(int i=0; i<int(v_samples.size()); i++)
		vIndex.push_back( predict(v_samples[i]) );

	return vIndex;
}

CSVM::t_confidence CSVM::predict_probability(const t_sample& sample)
{
	if(not m_b_trained)
		throw error("[CSVM::Write] First train the SVM.");

	int len  = int(sample.size());
	m_xspace = (struct svm_node *) realloc(m_xspace, (len+1)*sizeof(struct svm_node));

	for(int i=0; i<len; i++)
	{
		m_xspace[i].index = i+1;
		m_xspace[i].value = sample[i];
	}

	m_xspace[len].index = -1;

	t_confidence confidence;
	confidence.resize(m_model->nr_class, 0.0);
	svm_predict_probability(m_model, m_xspace, &confidence[0]);

	return confidence;
}

vector<CSVM::t_confidence> CSVM::predict_probability(const vector<t_sample>& v_samples)
{
	vector<t_confidence> v_prop;
	int nsample = int(v_samples.size());
	v_prop.resize(nsample);

	for(int i=0; i<nsample; i++)
		v_prop.push_back( predict_probability(v_samples[i]) );

	return v_prop;
}

void CSVM::read(const path& pthinput)
{
	if( (m_model=svm_load_model(pthinput)) == 0 )
		throw error("[CSVM::Read] SVM file can't open., " + pthinput);

	m_b_trained = true;
}

void CSVM::write(const path& pthoutput)
{
	if(not m_b_trained)
		throw error("[CSVM::Write] First train the SVM.");

	if(svm_save_model(pthoutput, m_model))
		throw error("[CSVM::Write] SVM file can't open., " + pthoutput);
}
