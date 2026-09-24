#!/usr/bin/env python3
"""Render the seven main-text and three supplementary figures, without fitting.

Inputs: the versioned current manuscript data bundle (analysis.json, rows.parquet,
coverage.parquet, and predictions/*.npz). All positions are physical table row
positions, not DataFrame index labels. Units and samples match the manuscript.
Run: python reproduce/current/figures.py --data-dir PATH --output-dir PATH
Outputs: PDF/PNG pairs plus figure_sources.json containing input SHA-256 hashes.
This is a rendering stage; model fitting and explanation computation are separate.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import set_jgr, finish


def metrics(y, p):
    """Calculate plotted metrics directly from saved predictions."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return dict(n=len(y), MAE=float(np.abs(p-y).mean()),
                r=float(np.corrcoef(y, p)[0, 1]))


def label(c):
    """Use the final manuscript's human-readable feature labels."""
    c = c.replace('newell_cf','Coupling').replace('dipole_tilt','Dipole tilt').replace('imf_','IMF ').replace('sw_pdyn','Pressure').replace('sw_n','Density').replace('sw_v','Speed').replace('by_hemi','By × hemisphere').replace('hemi_code','Hemisphere').replace('doy','Day of year')
    return re.sub(r'_(mean|std|delta|int)(\d+)', r' \1 (\2 min)', c)


def scatter(ax, y, p, title, unit='deg'):
    """Plot observed and predicted labels for a documented test sample."""
    h = ax.hexbin(y,p,gridsize=45,mincnt=1,bins='log',cmap='viridis',rasterized=True)
    lo=min(y.min(),p.min()); hi=max(y.max(),p.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=1)
    ax.set(xlabel=f'Observed [{unit}]',ylabel=f'Predicted [{unit}]',title=title)
    m=metrics(y,p)
    ax.text(.04,.96,f'MAE = {m["MAE"]:.2f} {unit}\nr = {m["r"]:.3f}',transform=ax.transAxes,va='top',fontsize=11)
    return h


class Renderer:
    """Read validated immutable sources and write figures to an explicit directory."""
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.sources = set()
        self.products = []
        self.result = json.loads(self.source('analysis.json').read_text())
        self.frame = pd.read_parquet(self.source('rows.parquet'))
        self.indices = np.flatnonzero(self.frame.year.to_numpy() >= 2008)
        if len(self.indices) != self.result['n_test']:
            raise ValueError('Temporal test count differs from analysis.json')
        if len(self.frame) - len(self.indices) != self.result['n_train']:
            raise ValueError('Temporal training count differs from analysis.json')
        self.observed, self.predicted = self.prediction('primary', 'abs_eq_mlat')
        self.check(self.result['primary'], self.observed, self.predicted)

    def source(self, relative):
        path = self.data_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing bundle input: {path}')
        self.sources.add(relative)
        return path

    def prediction(self, name, target):
        """Reject misaligned prediction caches instead of plotting them silently."""
        with np.load(self.source(f'predictions/{name}.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['test_indices'], self.indices)
            observed = z['observed'].astype(float)
            predicted = z['predicted'].astype(float)
        expected = self.frame.iloc[self.indices][target].to_numpy(np.float32).astype(float)
        np.testing.assert_array_equal(observed, expected)
        if predicted.shape != observed.shape or not np.isfinite(predicted).all():
            raise ValueError(f'Invalid prediction vector: {name}')
        return observed, predicted

    @staticmethod
    def check(record, observed, predicted):
        actual = metrics(observed, predicted)
        if record['n'] != actual['n'] or abs(record['MAE'] - actual['MAE']) > 1e-6:
            raise ValueError('Stored MAE/count differs from prediction cache')

    def save(self, fig, name, dpi=180, style=True):
        if style:
            for ax in fig.axes:
                finish(ax, minor_x=ax.get_xscale() == 'linear',
                       minor_y=ax.get_yscale() == 'linear')
        for suffix in ('.pdf', '.png'):
            fig.savefig(self.output_dir / (name + suffix), bbox_inches='tight', dpi=dpi)
        self.products.append(name)
        plt.close(fig)

    def scatter_figure(self):
        """Render the documented manuscript figure from checked inputs."""
        R=self.result; yt=self.observed; p=self.predicted; e=p-yt
        figsave=self.save
        fig,ax=plt.subplots(figsize=(5.5,4.6));h=scatter(ax,yt,p,'Equatorward boundary latitude');fig.colorbar(h,ax=ax,label='Crossings per bin');figsave(fig,'temporal_scatter')


    def errors(self):
        """Render the documented manuscript figure from checked inputs."""
        R=self.result; yt=self.observed; p=self.predicted; e=p-yt
        figsave=self.save
        fig,axes=plt.subplots(1,2,figsize=(8.2,3.2));axes[0].hist(e,bins=np.linspace(-8,8,65),color='#173f5f');axes[0].axvline(0,color='k',ls='--');axes[0].set(xlabel='Predicted minus observed [deg]',ylabel='Crossings')
        xx=np.sort(abs(e));axes[1].plot(xx,np.arange(1,len(xx)+1)/len(xx),color='#173f5f');axes[1].set(xlabel='Absolute error [deg]',ylabel='Cumulative fraction',xlim=(0,5),ylim=(0,1.02));figsave(fig,'temporal_errors')


    def explanations(self):
        """Render the documented manuscript figure from checked inputs."""
        R=self.result; save=self.save
        fig,axes=plt.subplots(2,1,figsize=(6.8,6.6))
        for ax,k,xlab in zip(axes,['gain','shap'],['Fraction of total gain','Mean absolute SHAP [deg]']):
         items=list(R[k].items())[:8][::-1];ax.barh([label(k) for k,v in items],[v for k,v in items],color='#173f5f');ax.set_xlabel(xlab);ax.tick_params(axis='y',labelsize=11)
        fig.tight_layout(h_pad=1.5);save(fig,'temporal_importance')
        # Six panels in two columns preserve labels at journal text width.
        fig,axes=plt.subplots(3,2,figsize=(7.0,7.0))
        units=['deg','coupling units','nT','nPa','nT','km/s nT']
        for ax,(k,d),u in zip(axes.flat,R['pdp'].items(),units):
         ax.plot(d['grid'],d['prediction'],color='#173f5f');ax.set(xlabel={'newell_cf_mean60':'60-min coupling [a.u.]','vBs_mean60':'60-min vBs [km/s nT]'}.get(k,label(k)+' ['+u+']'),ylabel='Mean prediction [deg]')
        fig.tight_layout();save(fig,'temporal_pdp')


    def residuals(self):
        """Render the documented manuscript figure from checked inputs."""
        R=self.result; yt=self.observed; p=self.predicted; e=p-yt
        figsave=self.save
        testdf=self.frame.iloc[self.indices];ae=testdf.ae_index.to_numpy(float)
        fig,axes=plt.subplots(2,2,figsize=(8.4,6));computed={}
        for ax,(name,z) in zip(axes.flat,[('Predicted latitude [deg]',p),('60-min mean coupling',testdf.newell_cf_mean60.to_numpy()),('Dipole tilt [deg]',testdf.dipole_tilt.to_numpy()),('AE index [nT]',ae)]):
         ax.scatter(z,e,s=2,alpha=.12,color='#173f5f',rasterized=True);edges=np.unique(np.quantile(z[np.isfinite(z)],[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]));mid=[];med=[]
         for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
          b=(z>=lo)&((z<=hi) if k==len(edges)-2 else (z<hi));mid.append(float(np.median(z[b])));med.append(float(np.median(e[b])))
         ax.plot(mid,med,'o-',ms=3,color='#b03030',label='Bin median');ax.axhline(0,color='k',ls='--',lw=.8);ax.set(xlabel=name,ylabel='Residual [deg]',ylim=(-7,7));ax.legend(fontsize=9);computed[name]={'x':mid,'median':med}
        fig.tight_layout();figsave(fig,'temporal_residuals')


    def other_targets(self):
        """Render the documented manuscript figure from checked inputs."""
        fig,axes=plt.subplots(1,3,figsize=(11.2,3.6))
        for ax,target,title,unit in zip(axes,['abs_pole_mlat','eq_mlt','mean_mlt'],['Poleward boundary latitude','Equatorward boundary MLT','Mean crossing MLT'],['deg','hr','hr']):
            y,p=self.prediction(target,target)
            self.check(self.result['other_targets'][target],y,p)
            scatter(ax,y,p,title,unit)
        fig.tight_layout();self.save(fig,'temporal_other_targets')


    def baselines(self):
        """Render the documented manuscript figure from checked inputs."""
        result=self.result
        rows = [
            ('Newell (2006) form', 'Newell (2006)\ncoupling function'),
            ('MLP74', 'MLP (74 features)'),
            ('Ridge74', 'Ridge (74 features)'),
            ('GBR300', 'GBR (74 features)'),
            ('XGBoost74', 'XGBoost (this study)\n74 features'),
        ]
        assert result['n_test'] == 9733
        values = []
        for key, _ in rows:
            metrics = result['models'][key]
            assert metrics['n'] == result['n_test']
            values.append(metrics['MAE'])
        assert values == sorted(values, reverse=True), 'Expected descending MAE order'
        set_jgr(12)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        positions = list(range(len(rows)))
        ax.barh(positions, values, height=0.58,
                color=['#999999', '#a08a70', '#8d9d9e', '#74949a', '#173f5f'])
        ax.set_yticks(positions, [label for _, label in rows])
        ax.invert_yaxis()
        ax.set(xlabel='Mean absolute error [deg]', xlim=(0, 2.02))
        for position, value in zip(positions, values):
            ax.text(value + 0.035, position, f'{value:.2f}', va='center', fontsize=12)
        ax.get_yticklabels()[-1].set_fontweight('bold')
        finish(ax, minor_y=False)
        fig.tight_layout()
        self.save(fig,'temporal_baseline',style=False)


    def windows(self):
        """Render the documented manuscript figure from checked inputs."""
        data=self.result
        windows = [0, 15, 30, 60, 90, 120]
        rows = [dict(window=w, **data['windows'][str(w)]) for w in windows]
        assert all(r['n'] == data['n_test'] for r in rows)
        set_jgr(12)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig, ax = plt.subplots(figsize=(6.8, 3.5))
        values = [r['MAE'] for r in rows]
        ax.plot(windows, values, 'o-', color='#173f5f', lw=1.5, ms=6)
        for r in rows:
            label = f"{r['n_features']} features"
            if r['window'] == 60:
                label += "\nMain model"
            ax.annotate(label, (r['window'], r['MAE']),
                        xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
        reference = data['windows']['60']['MAE']
        extra90 = reference - data['windows']['90']['MAE']
        extra120 = reference - data['windows']['120']['MAE']
        ax.scatter([60], [reference], s=65, color='#b66a22', zorder=4)
        ax.text(0.97, 0.92,
                'Additional MAE reduction vs. 60 min\n'
                f'90 min: {extra90:.2f} deg; 120 min: {extra120:.2f} deg',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#cccccc'))
        ax.set(xlabel='Maximum history window included [min]',
               ylabel='Mean absolute error [deg]', xticks=windows,
               xlim=(-10, 131), ylim=(1.04, 1.54),
               title='Cumulative history features: modest gains beyond 60 min')
        finish(ax)
        fig.tight_layout()
        self.save(fig,'temporal_windows',style=False)


    def hemisphere(self):
        """Render the documented manuscript figure from checked inputs."""
        test=self.frame.iloc[self.indices];obs=self.observed;pred=self.predicted;r=self.result
        set_jgr(12);plt.rcParams['font.family']='DejaVu Sans'
        fig=plt.figure(figsize=(7.4,6.4));gs=fig.add_gridspec(2,2)
        axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
        verified={}
        for ax,h in zip(axes[:2],['N','S']):
            m=test.hemisphere.to_numpy()==h;y=obs[m];q=pred[m]
            count=int(m.sum());mae=float(abs(q-y).mean())
            assert count==r['hemisphere'][h]['n']
            assert abs(mae-r['hemisphere'][h]['MAE'])<1e-6
            verified[h]=dict(test_crossings=count,MAE=mae)
            ax.hexbin(y,q,gridsize=32,mincnt=1,bins='log',cmap='viridis',rasterized=True)
            lo=min(y.min(),q.min());hi=max(y.max(),q.max());ax.plot([lo,hi],[lo,hi],'k--',lw=1)
            ax.set(xlabel='Observed [deg]',ylabel='Predicted [deg]',title='Northern Hemisphere' if h=='N' else 'Southern Hemisphere')
            ax.text(.04,.97,f'MAE = {mae:.2f} deg\nTest crossings: {count:,}',transform=ax.transAxes,
                    va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
        bins=[0,100,300,500,np.inf];activity=[];ae=test.ae_index.to_numpy()
        for i,(low,high) in enumerate(zip(bins[:-1],bins[1:])):
            m=(ae>=low)&(ae<high);count=int(m.sum());mae=float(abs(pred[m]-obs[m]).mean())
            assert count==r['activity'][i]['n'];assert abs(mae-r['activity'][i]['MAE'])<1e-6
            activity.append(dict(test_crossings=count,MAE=mae))
        assert sum(v['test_crossings'] for v in activity)==len(test)
        ax=axes[2];ax.bar(range(4),[v['MAE'] for v in activity],color='#173f5f')
        ax.set(xticks=range(4),xticklabels=['0 to <100','100 to <300','300 to <500','500 or more'],
               xlabel='AE index [nT]',ylabel='Mean absolute error [deg]',ylim=(0,1.95))
        for i,v in enumerate(activity):
            ax.text(i,v['MAE']+.04,f"{v['test_crossings']:,}\ntest crossings",ha='center',fontsize=10)
        for ax in axes:finish(ax)
        fig.suptitle('Temporal test set: 9,733 crossings (2008 to 2014)',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.96))
        self.save(fig,'temporal_hemisphere',style=False)


    def coverage(self):
        """Render the documented manuscript figure from checked inputs."""
        df=pd.read_parquet(self.source('coverage.parquet'))
        if 'year' not in df:
            df['year']=pd.to_datetime(df['time_start']).dt.year
        df['hemi_code']=(df['hemisphere']=='N').astype(int)
        assert len(df) == 40813
        assert df['hemisphere'].value_counts().to_dict() == {'N': 36805, 'S': 4008}
        fig=plt.figure(figsize=(15,5))
        gs=fig.add_gridspec(1,3,width_ratios=[1.5,1,1])

        # (a) stacked bar
        ax=fig.add_subplot(gs[0,0])
        sats=sorted(df['satellite'].unique()); cmap=plt.cm.tab20
        cols={s:cmap(i/max(len(sats)-1,1)) for i,s in enumerate(sats)}
        years=np.arange(df['year'].min(),df['year'].max()+1); bottom=np.zeros(len(years))
        for s in sats:
            c=df[df['satellite']==s].groupby('year').size().reindex(years,fill_value=0).values
            ax.bar(years,c,bottom=bottom,label=s,color=cols[s],width=0.8,edgecolor='none'); bottom+=c
        ax.set_xlabel('Year'); ax.set_ylabel('Number of Crossings')
        ax.set_title('(a) Crossings by Year and Satellite')
        ax.legend(ncol=3,fontsize=8,loc='upper left',framealpha=0.9); ax.set_xlim(1986.5,2014.5)
        # Stacked bars pin sticky edges at every segment bottom, which lets the
        # tallest year (2003, 4,251 crossings) touch the axes frame; add headroom.
        ax.set_ylim(0, float(bottom.max())*1.08)

        # (b)/(c) full-circle pole-view dials
        def dial(ax_p, mask, title, color):
            theta=((df['eq_mlt'].values[mask]-12)/24.0)*2*np.pi
            r=90-df['eq_mlat'].abs().values[mask]
            ax_p.scatter(theta,r,s=2,alpha=0.35,c=color,rasterized=True)
            ax_p.set_theta_zero_location('N'); ax_p.set_theta_direction(1)
            ax_p.set_thetamin(-180); ax_p.set_thetamax(180)          # force FULL circle
            ax_p.set_rlim(0,30); ax_p.set_rticks([5,10,15,20,25])
            ax_p.set_yticklabels(['85°','80°','75°','70°','65°'],fontsize=9)
            ax_p.set_xticks(((np.array([0,6,12,18])-12)/24.0)*2*np.pi)
            ax_p.set_xticklabels(['00','06','12','18'],fontsize=11)
            ax_p.set_title(f'{title}\n{int(mask.sum()):,} crossings',pad=16)

        dial(fig.add_subplot(gs[0,1],projection='polar'),(df['hemi_code'].values==1),'(b) Northern Hemisphere','steelblue')
        dial(fig.add_subplot(gs[0,2],projection='polar'),(df['hemi_code'].values==0),'(c) Southern Hemisphere','indianred')

        plt.tight_layout()
        self.save(fig,'fig01_data_coverage',dpi=120,style=False)


    def run(self):
        """Render all ten figures and write machine-readable input provenance."""
        # Coverage uses its original independent style; temporal figures share JGR style.
        with plt.rc_context({'font.size':12,'axes.labelsize':13,'axes.titlesize':13,
                             'xtick.labelsize':11,'ytick.labelsize':11}):
            self.coverage()
        set_jgr(12)
        plt.rcParams['font.family']='DejaVu Sans'
        for draw in (self.scatter_figure,self.baselines,self.errors,self.explanations,
                     self.hemisphere,self.residuals,self.other_targets,self.windows):
            draw()
        hashes={name:hashlib.sha256((self.data_dir/name).read_bytes()).hexdigest()
                for name in sorted(self.sources)}
        (self.output_dir/'figure_sources.json').write_text(json.dumps({
            'inputs_sha256':hashes,'figures':self.products,
            'n_train':self.result['n_train'],'n_test':self.result['n_test'],
            'stage':'render from archived predictions and explanation arrays; no model fitting'},indent=2)+'\n')
        print(f'Rendered {len(self.products)} manuscript figures in {self.output_dir}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    Renderer(args.data_dir,args.output_dir).run()


if __name__=='__main__':
    main()
