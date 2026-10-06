"""Independent repeat experiments estimate the ellipses used by a camera clock fit."""
from pathlib import Path
import json
import numpy as np
import clock_model as clock
import optics

SEED = 20261006
REPEATS = (4, 8, 32, 128)


def sample_covariance(repeats):
    """Shape (events, repeats, 2); subtract each event's unknown repeat mean.

    The repeats must share a fixed true timestamp pair within each event.
    Their mean is deliberately NOT used as a synchronization observation.
    """
    z = np.asarray(repeats, dtype=float)
    if z.ndim != 3 or z.shape[2] != 2 or z.shape[1] < 3 or not np.all(np.isfinite(z)):
        raise ValueError('Require finite (events, m>=3, 2) fixed-event repeats')
    residual = z - z.mean(axis=1, keepdims=True)
    c = np.einsum('nki,nkj->nij', residual, residual) / (z.shape[1]-1)
    if np.any(np.linalg.eigvalsh(c)[:, 0] <= 0):
        raise ValueError('Repeat covariance is singular; obtain more independent repeats')
    return c


def shrink_correlation(c, strength=.25):
    """Fixed heuristic: retain marginal variances, shrink off-diagonal terms.

    Not a fitted optimum or a correction for covariance estimation uncertainty.
    """
    c = np.asarray(c, dtype=float)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Shrinkage must lie in [0,1]')
    if c.ndim != 3 or c.shape[1:] != (2,2) or not np.all(np.isfinite(c)):
        raise ValueError('Require finite covariance array')
    if not np.allclose(c,c.transpose(0,2,1)) or np.any(np.linalg.eigvalsh(c)[:,0]<=0):
        raise ValueError('Require symmetric positive definite covariances')
    out=c.copy()
    out[:,0,1] *= 1-strength
    out[:,1,0] *= 1-strength
    return out


def summarize(errors):
    """All errors in milliseconds; failures are counted separately, never hidden."""
    e=np.asarray(errors, float)
    return dict(bias_ms=float(e.mean()), sd_ms=float(e.std(ddof=1)),
                rmse_ms=float(np.sqrt(np.mean(e**2))),
                mean_error_mcse_ms=float(e.std(ddof=1)/np.sqrt(len(e))),
                p95_absolute_error_ms=float(np.quantile(abs(e),.95)), successful_trials=len(e))


def demo(out='outputs', trials=800):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Independent streams prevent repeat-budget changes from changing camera/sync data.
    image_rng, sync_rng, repeat_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(3)]
    c=clock.covariances(8); chol=np.linalg.cholesky(c)
    truth=np.column_stack((clock.SYNC,clock.RATE*clock.SYNC+clock.OFFSET))
    center=optics.ORIGIN+optics.VELOCITY*(clock.RATE*(clock.FRAME+clock.DELAY)+clock.OFFSET)
    centers=np.array([optics.fit_center(optics.image(center)+image_rng.normal(0,optics.READ_SD,len(optics.PIXELS))) for _ in range(trials)])
    observations=truth+np.einsum('nij,tnj->tni',chol,sync_rng.normal(size=(trials,len(c),2)))
    oracle=np.array([1000*(clock.delay_from_center(u,clock.calibrate(*z.T,c))-clock.DELAY) for u,z in zip(centers,observations)])
    errors={m:{'estimated':[], 'shrunk':[]} for m in REPEATS}
    covariance_errors={m:[] for m in REPEATS}
    failures={m:{'estimated':0,'shrunk':0} for m in REPEATS}
    # Generate a fresh independent repeat calibration for EVERY outer experiment.
    # Prefixes pair the different repeat budgets; no truth covariance reaches fits.
    for u,z in zip(centers,observations):
        repeats=truth[:,None,:]+np.einsum('nij,nkj->nki',chol,repeat_rng.normal(size=(len(c),max(REPEATS),2)))
        for m in REPEATS:
            estimated=sample_covariance(repeats[:,:m])
            covariance_errors[m].append(float(np.mean(((estimated[:,0,0]/c[:,0,0])-1)**2)))
            for name,assumed in [('estimated',estimated),('shrunk',shrink_correlation(estimated))]:
                try:
                    error=1000*(clock.delay_from_center(u,clock.calibrate(*z.T,assumed))-clock.DELAY)
                    errors[m][name].append(error)
                except ValueError:
                    failures[m][name]+=1
    rows=[]
    for m in REPEATS:
        row=dict(repeats_per_event=m, relative_variance_mse=float(np.mean(covariance_errors[m])),
                 wishart_prediction=2/(m-1),oracle=summarize(oracle))
        for name in errors[m]:
            row[name]=summarize(errors[m][name]);row[name]['failed_trials']=failures[m][name]
        rows.append(row)
    report=dict(seed=SEED,trials=trials,events=40,heterogeneity=8,shrinkage=.25,
                sigma_px=optics.SIGMA,center_sd_px=float(centers.std(ddof=1)),
                true_delay_ms=1000*clock.DELAY,
                noiseless_delay_ms=float(1000*clock.delay_from_center(center,clock.calibrate(*truth.T,c))),
                experiment=rows)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ax[0].plot(optics.PIXELS,optics.image(center)-optics.BACKGROUND,'o-')
    ax[0].set(xlim=(-1,5),xlabel='Sensor coordinate (px)',ylabel='Integrated signal (ADU)',title='Diffraction -> pixels -> optical time')
    ax[1].loglog(REPEATS,[r['relative_variance_mse'] for r in rows],'o-',label='Independent repeat simulation')
    ax[1].loglog(REPEATS,[r['wishart_prediction'] for r in rows],'--',label='Analytic 2 / (m - 1)')
    ax[1].set(xlabel='Independent repeats per event m',ylabel='Relative variance-estimate MSE',title='Uncertain covariance is measurable');ax[1].legend(fontsize=8)
    for name,label in [('oracle','True covariance (oracle)'),('estimated','Estimated covariance'),('shrunk','25% correlation shrinkage')]:
        ax[2].semilogx(REPEATS,[r[name]['rmse_ms'] for r in rows],'o-',label=label)
    ax[2].set(xlabel='Independent repeats per event m',ylabel='Camera-delay RMSE (ms)',title='More repeats stabilize optical delay');ax[2].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))
    if any(sum(f.values()) for f in failures.values()):
        raise RuntimeError('Some fits failed; RMSE is conditional on success; inspect JSON')

if __name__=='__main__':demo()
