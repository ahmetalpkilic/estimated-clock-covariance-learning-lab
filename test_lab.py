import unittest
import numpy as np
from scipy.optimize import least_squares
import lab
import clock_model as c

class CovarianceTests(unittest.TestCase):
    def test_independent_numpy_cov_and_shift(self):
        rng=np.random.default_rng(70);z=rng.normal(size=(5,12,2))
        expected=np.array([np.cov(zi,rowvar=False,ddof=1) for zi in z])
        np.testing.assert_allclose(lab.sample_covariance(z),expected,atol=1e-14)
        np.testing.assert_allclose(lab.sample_covariance(z+np.arange(10).reshape(5,1,2)),expected,atol=1e-14)
        np.testing.assert_allclose(lab.sample_covariance(z*1000),expected*1e6,rtol=1e-13)

    def test_wishart_moments_independent_seed(self):
        # Independent 50,000 replicate experiments validate denominator and cross terms.
        rng=np.random.default_rng(120);truth=np.array([[4.,1.2],[1.2,1.]])
        z=rng.normal(size=(50000,8,2))@np.linalg.cholesky(truth).T
        s=lab.sample_covariance(z)
        np.testing.assert_allclose(s.mean(axis=0),truth,rtol=.012)
        predicted=(truth[0,0]*truth[1,1]+truth[0,1]**2)/7
        self.assertLess(abs(s[:,0,1].var()-predicted)/predicted,.025)
        self.assertLess(abs(np.mean((s[:,0,0]/4-1)**2)-2/7),.006)

    def test_shrinkage_and_invalid(self):
        noise=c.covariances(8)
        np.testing.assert_allclose(lab.shrink_correlation(noise,0),noise)
        diagonal=lab.shrink_correlation(noise,1)
        np.testing.assert_allclose(diagonal[:,0,1],0)
        np.testing.assert_allclose(np.diagonal(diagonal,axis1=1,axis2=2),np.diagonal(noise,axis1=1,axis2=2))
        for z in [np.ones((5,2,2)),np.ones((5,4,2)),np.full((5,4,2),np.nan),np.ones((5,4,3))]:
            with self.assertRaises(ValueError):lab.sample_covariance(z)
        for strength in [-.1,1.1,np.nan]:
            with self.assertRaises(ValueError):lab.shrink_correlation(noise,strength)

    def test_repeat_budget_reduces_delay_error(self):
        # Independent seed, identical evaluation pairs, exact optical center.
        # This isolates repeat-estimation variance from camera localization noise.
        import optics
        rng=np.random.default_rng(341)
        noise=c.covariances(8);chol=np.linalg.cholesky(noise)
        truth=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
        center=optics.ORIGIN+optics.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
        errors=[]
        for _ in range(250):
            z=truth+np.einsum('nij,nj->ni',chol,rng.normal(size=(40,2)))
            repeats=truth[:,None,:]+np.einsum('nij,nkj->nki',chol,rng.normal(size=(40,128,2)))
            errors.append([c.delay_from_center(center,c.calibrate(*z.T,lab.sample_covariance(repeats[:,:m])))-c.DELAY for m in [4,128]])
        mse=np.mean(np.array(errors)**2,axis=0)
        self.assertLess(mse[1],.7*mse[0])

    def test_estimated_covariance_joint_latent_fit(self):
        rng=np.random.default_rng(190);noise=c.covariances(8)
        truth=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
        repeats=truth[:,None,:]+np.einsum('nij,nkj->nki',np.linalg.cholesky(noise),rng.normal(size=(40,4,2)))
        estimated=lab.sample_covariance(repeats)
        z=truth+np.einsum('nij,nj->ni',np.linalg.cholesky(noise),rng.normal(size=(40,2)))
        p=c.calibrate(*z.T,estimated);chol=np.linalg.cholesky(estimated)
        def residual(p):
            r=z-np.column_stack((p[2:],p[0]*p[2:]+p[1]))
            return np.linalg.solve(chol,r[...,None])[...,0].ravel()
        fit=least_squares(residual,np.r_[1.,0.,z[:,0]],xtol=1e-12,ftol=1e-12,gtol=1e-10)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(p,fit.x[:2],atol=5e-8)

if __name__=='__main__':unittest.main()
