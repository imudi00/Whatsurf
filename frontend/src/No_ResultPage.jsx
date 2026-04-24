import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './ResultPage.css';

const SUPABASE_URL = 'http://localhost:8000';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yc2N4aG95ZGxuY2pxcHR2am1jIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3Mzg4MjAxMywiZXhwIjoyMDg5NDU4MDEzfQ.N_T0rYbxFb_qPdM4KPr7qZX2YjG0IhlDd2Q50RwrzYY';

const DUMMY_CLUSTERS = [
  { 
    id: 1, cluster_label: 0, cluster_title: "정부-의료계 강대강 대치 심화", cluster_count: 1240, 
    cluster_summary: "정부의 의대 정원 2000명 증원 확정 발표 이후, 전공의들의 집단 사직과 의대생들의 휴학 신청이 이어지며 의료 공백이 현실화되고 있습니다.", 
    leading_sources: [{source_name: '중앙일보'}, {source_name: '연합뉴스'}, {source_name: 'KBS'}], 
    representative_comments: [
      {comment_id: 1, cmt_content: "환자들만 볼모로 잡는 처사입니다. 당장 복귀하세요.", cmt_emotion: "분노"},
      {comment_id: 2, cmt_content: "교육 여건 부재를 주장하는 의료계 입장도 이해갑니다.", cmt_emotion: "중립"}
    ] 
  },
  { 
    id: 2, cluster_label: 1, cluster_title: "응급실 운영난 및 시민 불안 확산", cluster_count: 850, 
    cluster_summary: "전공의 이탈이 장기화되면서 주요 상급종합병원의 응급실이 파행 운영되고 있습니다. 이른바 '응급실 뺑뺑이' 사례가 속출하며 시민들의 불안감이 커지고 있습니다.", 
    leading_sources: [{source_name: 'SBS'}, {source_name: '한겨레'}, {source_name: 'MBC'}], 
    representative_comments: [
      {comment_id: 3, cmt_content: "아이가 아플 때 갈 곳이 없을까봐 너무 불안해요.", cmt_emotion: "불안"}
    ] 
  },
  { 
    id: 3, cluster_label: 2, cluster_title: "나나냥나나냥나나냥", cluster_count: 850, 
    cluster_summary: "냐냐냥냐냥캬옹으르럴크르릉크냥야옹캬옹으르르를아르르르르를캬아아앜", 
    leading_sources: [{source_name: 'SBS'}, {source_name: '한겨레'}, {source_name: 'MBC'}], 
    representative_comments: [
      {comment_id: 3, cmt_content: "아이가 아플 때 갈 곳이 없을까봐 너무 불안해요.", cmt_emotion: "불안"}
    ] 
  }, {
    id: 4, cluster_label: 3, cluster_title: "나나냥나나냥나나냥", cluster_count: 850,
    cluster_summary: "냐냐냥냐냥캬옹으르럴크르릉크냥야옹캬옹으르르를아르르르르를캬아아앜",
    leading_sources: [{source_name: 'SBS'}, {source_name: '한겨레'}, {source_name: 'MBC'}],
    representative_comments: [
      {comment_id: 3, cmt_content: "아이가 아플 때 갈 곳이 없을까봐 너무 불안해요.", cmt_emotion: "불안"}
    ]
  }
];

const DUMMY_FEATURE_MAPS = {
  cluster_0: [
    { id: 'p1', label: '사건 원인 집중', type: 'primary', top: '35%', left: '30%', color: '#3b82f6', desc: '의료 인력 부족 현상 분석' },
    { id: 's1', label: '필수의료 붕괴', type: 'secondary', top: '15%', left: '20%', ratio: 0.8 },
    { id: 's2', label: '정부 책임론', type: 'secondary', top: '50%', left: '40%', ratio: 0.6 },
    { id: 's3', label: '의료계 내부 갈등', type: 'secondary', top: '70%', left: '20%', ratio: 0.4 },
  ], 
  cluster_1: [
    { id: 'p2', label: '시민 불안 집중', type: 'primary', top: '40%', left: '60%', color: '#10b981', desc: '응급실 운영난과 시민 불안 분석' },
  ], 
  cluster_2: [
    { id: 'p3', label: '의료계 내부 갈등', type: 'primary', top: '50%', left: '50%', color: '#ec4899', desc: '정부와 의료계 간 갈등 분석' },
  ],
  cluster_3: [
    { id: 'p4', label: '정책적 대응 분석', type: 'primary', top: '30%', left: '70%', color: '#f59e0b', desc: '정부의 정책 대응과 향후 전망 분석' },
  ]
};

const DUMMY_MEDIA_DIST = [
  { id: 1, source_name: '조선일보', center_x: 0.85, center_y: 0.6, spread_radius: 0.15 },
  { id: 2, source_name: '한겨레', center_x: -0.8, center_y: 0.7, spread_radius: 0.2 },
  { id: 3, source_name: '중앙일보', center_x: 0.2, center_y: 0.3, spread_radius: 0.1 },
  { id: 4, source_name: 'MBC', center_x: -0.5, center_y: -0.4, spread_radius: 0.25 },
];

export default function ResultPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const searchTerm = location.state?.searchTerm || "의대 증원 논란";

  const [activeTab, setActiveTab] = useState('summary');
  const [mapSubTab, setMapSubTab] = useState('2d');
  const [selectedCluster, setSelectedCluster] = useState(0); 
  const [activeFeature, setActiveFeature] = useState(null);
  const [commentPlatform, setCommentPlatform] = useState('naver'); 
  const [isLoading, setIsLoading] = useState(true);

  const [totalArticles, setTotalArticles] = useState(0);
  const [clusterDetails, setClusterDetails] = useState([]);
  const [timelineData, setTimelineData] = useState([]);
  const [activeTimeline, setActiveTimeline] = useState(null);
  const [distributionData, setDistributionData] = useState([]);
  
  // X, Y 타겟 슬라이더 상태
  const [biasX, setBiasX] = useState(50);
  const [biasY, setBiasY] = useState(50);
  const [matchedArticle, setMatchedArticle] = useState(null);

  const currentNodes = DUMMY_FEATURE_MAPS[`cluster_${selectedCluster}`] || [];

  useEffect(() => {
    let isMounted = true;

    const loadTestData = async () => {
      setIsLoading(true);
      await new Promise(r => setTimeout(r, 500));
      if (!isMounted) return;

      setClusterDetails(DUMMY_CLUSTERS);
      const total = DUMMY_CLUSTERS.reduce((sum, c) => sum + (c.cluster_count || 0), 0);
      setTotalArticles(total);

      const dummyTimeline = [
        { id: 1, date: '02.06', volume: 20, phase: 'Phase 1', title: '의대 증원 발표', desc: '정부 공식 브리핑.', url: '#' },
        { id: 2, date: '02.20', volume: 100, phase: 'Phase 2', title: '전공의 사직', desc: '의료 대란 본격화.', url: '#' },
        { id: 3, date: '02.28', volume: 150, phase: 'Phase 3', title: '의료 인력 확보 방안 논의', desc: '정부와 의료기관 간 협의.', url: '#' },
        { id: 4, date: '03.10', volume: 80, phase: 'Phase 4', title: '응급실 운영난 심화', desc: '일부 병원 응급실 폐쇄.', url: '#' },
        { id: 5, date: '03.25', volume: 120, phase: 'Phase 5', title: '정부-의료계 협상 타결', desc: '임시 합의안 발표.', url: '#' },
        { id: 6, date: '04.05', volume: 90, phase: 'Phase 6', title: '의료 공백 완화 조치 시행', desc: '응급 인력 지원 및 임시 병상 확보.', url: '#' },
        { id: 7, date: '04.20', volume: 110, phase: 'Phase 7', title: '사태 진정세', desc: '의료계 점진적 복귀 및 안정화.', url: '#' }
      ];
      setTimelineData(dummyTimeline);
      setActiveTimeline(dummyTimeline[1]);

      const colors = ['#ec4899', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6'];
      const mappedDist = DUMMY_MEDIA_DIST.map((item, idx) => ({
        ...item,
        left: ((item.center_x + 1) / 2) * 100,
        top: ((1 - item.center_y) / 2) * 100,
        size: 80 + (item.spread_radius * 400),
        color: colors[idx % colors.length]
      }));
      setDistributionData(mappedDist);
      
      setIsLoading(false);
    };

    loadTestData();
    return () => { isMounted = false; };
  }, [searchTerm]);

  // ★ 부활한 매칭 로직! 유클리드 거리(피타고라스 정리)로 가장 가까운 점 찾기
  const handleBiasSearch = () => {
    if (distributionData.length === 0) return;

    let closest = null;
    let minDistance = Infinity;

    // Y축은 CSS top 값 기준이므로 100에서 빼줍니다 (슬라이더 100 = 화면 맨 위 0%)
    const targetY = 100 - biasY; 

    distributionData.forEach(media => {
      const dx = media.left - biasX;
      // 1D 모드일 때는 Y축 거리를 무시하고 X축(좌우) 거리만 계산합니다.
      const dy = mapSubTab === '2d' ? (media.top - targetY) : 0; 
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < minDistance) {
        minDistance = distance;
        closest = media;
      }
    });

    if (closest) {
      setMatchedArticle({ 
        press: closest.source_name, 
        title: `[심층분석] ${searchTerm} 사태, 핵심 쟁점과 전망`, // 가상의 기사 제목
        url: 'https://news.naver.com' // 가상의 기사 링크
      });
    }
  };

  return (
    <div className="whatsurf-container editorial-result">
      <div className="mesh-gradient-bg"></div>

      <header className="header">
        <div className="logo" onClick={() => navigate('/')} style={{cursor:'pointer'}}>whatsurf.</div>
        <button className="new-search-btn" onClick={() => navigate('/')}>New Search ↗</button>
      </header>

      <main className="result-main">
        <section className="result-hero">
          <h1 className="mega-title">{searchTerm}</h1>
          <p className="result-meta">• {isLoading ? '더미 데이터 로드 중...' : `${totalArticles.toLocaleString()}개의 기사 기반 분석 (TEST)`}</p>
        </section>

        <nav className="result-tabs">
          <button className={`tab-btn ${activeTab === 'summary' ? 'active' : ''}`} onClick={() => setActiveTab('summary')}>기사 요약</button>
          <button className={`tab-btn ${activeTab === 'timeline' ? 'active' : ''}`} onClick={() => setActiveTab('timeline')}>타임 라인</button>
          <button className={`tab-btn ${activeTab === 'feature' ? 'active' : ''}`} onClick={() => setActiveTab('feature')}>피처맵</button>
          <button className={`tab-btn ${activeTab === 'map' ? 'active' : ''}`} onClick={() => setActiveTab('map')}>좌표 평면</button>
        </nav>

        <section className="tab-content-area">
          {isLoading ? (
            <div className="loading-container" style={{ textAlign: 'center', padding: '100px 0' }}>
              <div className="spinner"></div>
            </div>
          ) : (
            <>
              {/* 기사 요약 & 타임라인 & 피처맵 로직 (이전과 동일) */}
              {activeTab === 'summary' && (
                <div className="tab-summary fade-in">
                  <div className="summary-cards-container">
                    {clusterDetails.map((cluster, index) => (
                      <div className="editorial-summary-card" key={index}>
                        <div className="esc-left">
                          <div className="esc-header">
                            <span className={`esc-badge ${index === 0 ? 'primary' : 'secondary'}`}>Cluster {index + 1}</span>
                            <h3 className="esc-title">"{cluster.cluster_title}"</h3>
                          </div>
                          <div className="esc-body">
                            <p className="esc-paragraph"><span className="drop-cap">{cluster.cluster_summary?.charAt(0)}</span>{cluster.cluster_summary?.slice(1)}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'timeline' && activeTimeline && (
                <div className="tab-timeline fade-in">
                  <div className="glass-card editorial-timeline-card">
                    <div className="massive-bg-date">{activeTimeline.date.replace('.', ' / ')}</div>
                    <div className="timeline-grid-layout">
                      <div className="timeline-chart-area">
                        <div className="volume-bar-container">
                          {timelineData.map((item) => (
                            <div key={item.id} className={`v-bar-group ${activeTimeline.id === item.id ? 'active' : ''}`} onClick={() => setActiveTimeline(item)}>
                              <div className="v-bar-track"><div className="v-bar-fill" style={{ height: `${item.volume}%` }}></div></div>
                              <span className="v-bar-date">{item.date}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="timeline-detail-area">
                        <div className="timeline-info-box">
                          <h3 className="d-article-title">{activeTimeline.title}</h3>
                          <p className="d-press-desc">{activeTimeline.desc}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'feature' && (
                <div className="tab-feature fade-in">
                  <div className="map-sub-nav" style={{display:'flex', gap:'10px', marginBottom:'20px'}}>
                    {clusterDetails.map((c, i) => (
                      <button key={i} className={`sub-tab-btn ${selectedCluster === c.cluster_label ? 'active' : ''}`} onClick={() => setSelectedCluster(c.cluster_label)}>관점 {i+1} 분석</button>
                    ))}
                  </div>
                  <div className="strict-grid-layout">
                    <div className="glass-card feature-map-bg" style={{minHeight:'600px', position:'relative', overflow:'hidden'}}>
                      {currentNodes.map((node) => {
                        const dynamicSize = node.type === 'primary' ? '130px' : `${(node.ratio * 200) + 50}px`;
                        return (
                          <div key={node.id} className={`f-node ${node.type} ${activeFeature?.id === node.id ? 'active' : ''}`} 
                            style={{ 
                              top: node.top, left: node.left, width: dynamicSize, height: dynamicSize,
                              backgroundColor: node.type === 'primary' ? '#111' : 'rgba(255,255,255,0.95)',
                              border: `2px solid ${node.color || '#3b82f6'}`, color: node.type === 'primary' ? '#fff' : '#111'
                            }} onClick={() => setActiveFeature(node)}>
                            <span style={{fontWeight:'700'}}>{node.label}</span>
                          </div>
                        );
                      })}
                    </div>
                    <div className="glass-card insight-panel">
                      {activeFeature && <><h2 className="press-name" style={{color: activeFeature.color}}>{activeFeature.label}</h2><p className="press-desc">{activeFeature.desc}</p></>}
                    </div>
                  </div>
                </div>
              )}

              {/* ◼️ 탭 4: 부활한 좌표 평면 기능! */}
              {activeTab === 'map' && (
                <div className="tab-map fade-in">
                  <div className="map-sub-nav" style={{display:'flex', gap:'10px', marginBottom:'20px'}}>
                    <button className={`sub-tab-btn ${mapSubTab === '2d' ? 'active' : ''}`} onClick={() => setMapSubTab('2d')}>2D 좌표 평면</button>
                    <button className={`sub-tab-btn ${mapSubTab === '1d' ? 'active' : ''}`} onClick={() => setMapSubTab('1d')}>1D 선형 분포</button>
                  </div>
                  <div className="strict-grid-layout">
                    <div className="glass-card map-visual-container">
                      <div className="scatter-plot-area" style={{minHeight:'500px', position:'relative'}}>
                        <div className="plot-grid"></div><div className="plot-axis x-axis"></div>
                        {mapSubTab === '2d' && <div className="plot-axis y-axis"></div>}
                        
                        {/* 언론사 노드 렌더링 */}
                        {distributionData.map((media, idx) => (
                          <React.Fragment key={idx}>
                            <div className="dist-blob" style={{ 
                                left: `${media.left}%`, top: mapSubTab === '2d' ? `${media.top}%` : '50%', 
                                width: `${media.size}px`, height: `${media.size}px`, background: media.color, opacity: 0.6 
                            }}></div>
                            <span className="dist-text" style={{ 
                                left: `${media.left}%`, top: mapSubTab === '2d' ? `${media.top}%` : '50%' 
                            }}>{media.source_name}</span>
                          </React.Fragment>
                        ))}

                        {/* ★ 부활한 타겟 포인터(크로스헤어) */}
                        <div className="target-crosshair" style={{ 
                          left: `${biasX}%`, 
                          top: mapSubTab === '2d' ? `${100 - biasY}%` : '50%',
                        }}>
                          <div className="crosshair-center"></div>
                        </div>

                      </div>
                    </div>
                    <div className="right-sidebar">
                      <div className="glass-card slider-control-panel">
                        <h3 className="control-title">TARGET SETTING</h3>
                        
                        {/* X축 슬라이더 */}
                        <div className="aesthetic-slider-group">
                          <div className="slider-labels"><span>진보</span><span className="slider-value-pill">X: {biasX}</span><span>보수</span></div>
                          <input type="range" min="0" max="100" value={biasX} onChange={(e)=>setBiasX(Number(e.target.value))} className="aesthetic-slider" style={{'--val': `${biasX}%`, '--color': '#3b82f6'}} />
                        </div>

                        {/* ★ 부활한 Y축 슬라이더 */}
                        {mapSubTab === '2d' && (
                          <div className="aesthetic-slider-group" style={{marginTop: '20px'}}>
                            <div className="slider-labels"><span>감정</span><span className="slider-value-pill">Y: {biasY}</span><span>분석</span></div>
                            <input type="range" min="0" max="100" value={biasY} onChange={(e)=>setBiasY(Number(e.target.value))} className="aesthetic-slider" style={{'--val': `${biasY}%`, '--color': '#ec4899'}} />
                          </div>
                        )}

                        <button className="search-target-btn" onClick={handleBiasSearch} style={{marginTop: '30px'}}>성향 매칭 🔍</button>
                      </div>

                      <div className="glass-card insight-panel map-insight">
                        {matchedArticle ? (
                          <>
                            <span className="insight-badge">매칭 완료</span>
                            <h2 className="press-name">{matchedArticle.press}</h2>
                            <h3 className="article-title">{matchedArticle.title}</h3>
                            <button 
                              className="editorial-read-btn glass-btn" 
                              style={{marginTop: '15px'}} 
                              onClick={() => window.open(matchedArticle.url, '_blank')}
                            >
                              기사 원문 읽기 →
                            </button>
                          </>
                        ) : (
                          <p style={{color: '#999', marginTop: '20px'}}>슬라이더를 조절해 원하는 성향의 기사를 찾아보세요.</p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}