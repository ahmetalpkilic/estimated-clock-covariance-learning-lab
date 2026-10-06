import unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
import optics as o
import clock_model as l

class Tests(unittest.TestCase):
    def test_independent_pupil_fourier(self):
        rho=np.linspace(-8000,8000,65536,endpoint=False); step=rho[1]-rho[0]
        pupil=np.exp(-(rho/o.PUPIL_RADIUS_UM)**2)
        intensity=abs(np.fft.fftshift(np.fft.fft(pupil)))**2
        u=np.fft.fftshift(np.fft.fftfreq(len(rho),step))*o.WAVELENGTH_UM*o.FOCAL_UM/o.PITCH_UM
        sd=np.sqrt(np.sum(u*u*intensity)/intensity.sum())
        self.assertAlmostEqual(sd,o.SIGMA,places=12)

    def test_pixel_integration_flux_derivative(self):
        c=.27
        independent=np.array([quad(lambda x: np.exp(-.5*((x-c)/o.SIGMA)**2)/(o.SIGMA*np.sqrt(2*np.pi)),j-.5,j+.5,epsabs=1e-13)[0] for j in o.PIXELS])
        np.testing.assert_allclose(o.image(c)-o.BACKGROUND,o.FLUX*independent,atol=1e-10)
        self.assertAlmostEqual(sum(o.image(c)-o.BACKGROUND),o.FLUX,places=9)
        np.testing.assert_allclose(o.derivative(c),(o.image(c+1e-6)-o.image(c-1e-6))/2e-6,atol=2e-6)
        self.assertAlmostEqual(o.fit_center(o.image(c)),c,places=10)

    def test_noiseless_and_homogeneous_special_case(self):
        for h in [1.,4.,8.]:
            p=l.calibrate(l.SYNC,l.RATE*l.SYNC+l.OFFSET,l.covariances(h))
            np.testing.assert_allclose(p,[l.RATE,l.OFFSET],atol=2e-8)
            center=o.ORIGIN+o.VELOCITY*(l.RATE*(l.FRAME+l.DELAY)+l.OFFSET)
            self.assertAlmostEqual(l.delay_from_center(center,p),l.DELAY,places=7)
        # Independent homogeneous closed-form generalized eigenvalue solution.
        from scipy.linalg import eigh
        rng=np.random.default_rng(3);c=l.covariances(1)
        z=np.column_stack((l.SYNC,l.RATE*l.SYNC+l.OFFSET))+np.einsum('nij,nj->ni',np.linalg.cholesky(c),rng.normal(size=(len(c),2)))
        q=z-z.mean(axis=0);_,v=eigh(q.T@q,c[0]);rate=-v[0,0]/v[1,0]
        np.testing.assert_allclose(l.calibrate(*z.T,c),[rate,z[:,1].mean()-rate*z[:,0].mean()],atol=2e-8)

    def test_joint_latent_optimization(self):
        rng=np.random.default_rng(14)
        for h in [1.,2.,8.]:
            c=l.covariances(h);chol=np.linalg.cholesky(c)
            z=np.column_stack((l.SYNC,l.RATE*l.SYNC+l.OFFSET))+np.einsum('nij,nj->ni',chol,rng.normal(size=(len(c),2)))
            fitted=l.calibrate(*z.T,c)
            def residual(p):
                r=z-np.column_stack((p[2:],p[0]*p[2:]+p[1]))
                return np.linalg.solve(chol,r[...,None])[...,0].ravel()
            result=least_squares(residual,np.r_[.95,.0,z[:,0]],xtol=1e-12,ftol=1e-12,gtol=1e-10)
            self.assertTrue(result.success)
            np.testing.assert_allclose(result.x[:2],fitted,atol=4e-8)
            latent=l.latent_times(*z.T,*fitted,c)
            np.testing.assert_allclose(result.x[2:],latent,atol=4e-8)
            full=residual(np.r_[fitted,latent])
            self.assertAlmostEqual(full@full,float(l.profile(fitted[0],*z.T,c)[0]),places=9)

    def test_dense_search_permutation_and_translation(self):
        rng=np.random.default_rng(17);c=l.covariances(8)
        z=np.column_stack((l.SYNC,l.RATE*l.SYNC+l.OFFSET))+np.einsum('nij,nj->ni',np.linalg.cholesky(c),rng.normal(size=(len(c),2)))
        a,b=l.calibrate(*z.T,c)
        q=np.sum((z[:,1]-a*z[:,0]-b)**2/(a*a*c[:,0,0]-2*a*c[:,0,1]+c[:,1,1]))
        grid=np.linspace(*l.RATE_BOUNDS,10001)
        self.assertLessEqual(q,float(l.profile(grid,*z.T,c)[0].min())+1e-8)
        perm=rng.permutation(len(c))
        np.testing.assert_allclose(l.calibrate(*z[perm].T,c[perm]),[a,b],atol=2e-8)
        np.testing.assert_allclose(l.calibrate(z[:,0]+4,z[:,1]-2,c),[a,b-2-4*a],atol=2e-8)
        np.testing.assert_allclose(l.calibrate(*(z*1000).T,c*1e6),[a,b*1000],atol=2e-6)

    def test_precision_benefit_with_known_covariances(self):
        # Independent seed, camera-center held exact to isolate calibration.
        rng=np.random.default_rng(432);c=l.covariances(8)
        truth=np.column_stack((l.SYNC,l.RATE*l.SYNC+l.OFFSET))
        center=o.ORIGIN+o.VELOCITY*(l.RATE*(l.FRAME+l.DELAY)+l.OFFSET)
        errors=[]
        for _ in range(180):
            z=truth+np.einsum('nij,nj->ni',np.linalg.cholesky(c),rng.normal(size=(len(c),2)))
            errors.append([l.delay_from_center(center,l.calibrate(*z.T,assumed))-l.DELAY for assumed in [c,np.broadcast_to(c.mean(axis=0),c.shape)]])
        errors=np.array(errors)
        self.assertLess(np.mean(errors[:,0]**2),.2*np.mean(errors[:,1]**2))
        self.assertLess(abs(errors[:,0].mean()),4*errors[:,0].std(ddof=1)/np.sqrt(len(errors)))

    def test_invalid_and_boundary(self):
        c=l.covariances(2)
        for x,y,noise in [(l.SYNC,l.SYNC,c[:2]),(l.SYNC,l.SYNC,np.zeros_like(c)),(l.SYNC,l.SYNC,c+np.array([[0,1],[0,0]])),(l.SYNC,np.full(len(c),np.nan),c),(l.SYNC,l.SYNC*2,c),(np.ones(len(c)),l.SYNC,c)]:
            with self.assertRaises(ValueError):l.calibrate(x,y,noise)
        for h in [0,9,float('nan')]:
            with self.assertRaises(ValueError):l.covariances(h)

if __name__=='__main__':unittest.main()
