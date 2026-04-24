import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './ResultPage.css';

const SUPABASE_URL = 'http://localhost:8000';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yc2N4aG95ZGxuY2pxcHR2am1jIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3Mzg4MjAxMywiZXhwIjoyMDg5NDU4MDEzfQ.N_T0rYbxFb_qPdM4KPr7qZX2YjG0IhlDd2Q50RwrzYY';

// --- [강화된 더미데이터] ---
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
  }
];

const DUMMY_FEATURE_MAPS = {
  cluster_0: [
    { id: 'p1', label: '사건 원인 집중', type: 'primary', top: '35%', left: '30%', color: '#3b82f6', desc: '의료 인력 부족 현상 분석' },
    { id: 's1', label: '필수의료 붕괴', type: 'secondary', top: '15%', left: '20%', ratio: 0.8 },
  ]
};

const DUMMY_MEDIA_DIST = [
  { id: 1, source_name: '조선일보', center_x: 0.85, center_y: 0.6, spread_radius: 0.15 },
  { id: 2, source_name: '한겨레', center_x: -0.8, center_y: 0.7, spread_radius: 0.2 },
];

export default function ResultPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const searchTerm = location.state?.searchTerm || "의대 증원 논란";

  // 상태 관리 (타임라인 탭이 돌아왔습니다!)
  const [activeTab, setActiveTab] = useState('summary');
  const [mapSubTab, setMapSubTab] = useState('2d');
  const [selectedCluster, setSelectedCluster] = useState(0); 
  const [activeFeature, setActiveFeature] = useState(null);
  const [commentPlatform, setCommentPlatform] = useState('naver'); 
  const [isLoading, setIsLoading] = useState(true);

  // 데이터 상태
  const [queryId, setQueryId] = useState(null);
  const [totalArticles, setTotalArticles] = useState(0);
  const [clusterDetails, setClusterDetails] = useState([]);
  const [timelineData, setTimelineData] = useState([]);
  const [activeTimeline, setActiveTimeline] = useState(null);
  const [featureMapsData, setFeatureMapsData] = useState(null); 
  const [distributionData, setDistributionData] = useState([]);
  const [biasX, setBiasX] = useState(50);
  const [biasY, setBiasY] = useState(50);
  const [matchedArticle, setMatchedArticle] = useState(null);

  const currentNodes = featureMapsData ? (featureMapsData[selectedCluster] || []) : (DUMMY_FEATURE_MAPS[`cluster_${selectedCluster}`] || []);

  const fetchAPI = async (endpoint, method = 'GET', body = null) => {
    try {
      const options = {
        method, headers: { 'Content-Type': 'application/json', 'apikey': SUPABASE_KEY, 'Authorization': `Bearer ${SUPABASE_KEY}` }
      };
      if (body) options.body = JSON.stringify(body);
      const response = await fetch(`${SUPABASE_URL}${endpoint}`, options);
      if (!response.ok) throw new Error("Server Offline");
      return await response.json();
    } catch (err) {
      return null; 
    }
  };

  useEffect(() => {
    let isMounted = true;

    const fetchAllData = async () => {
      try {
        setIsLoading(true);
        console.log(`🔍 [검색 시작] "${searchTerm}" 서버 요청`);

        const postData = await fetchAPI('/api/queries', 'POST', { query_text: searchTerm });
        const currentQueryId = postData?.data?.id || postData?.id || "dummy-id";
        if (isMounted) setQueryId(currentQueryId);

        let clusters = [];
        const maxTries = 360;
        const intervalMs = 10000;

        for (let i = 1; i <= maxTries; i++) {
          if (!isMounted) return;
          console.log(`⏳ 데이터 대기 중... (시도: ${i}/${maxTries})`);

          const clusterData = await fetchAPI(`/api/queries/${currentQueryId}/clusters`);
          clusters = clusterData?.data?.clusters || clusterData?.clusters || [];

          if (clusters.length > 0) {
            console.log("✅ 데이터 수신 완료!");
            break;
          }
          if (i < maxTries) await new Promise(r => setTimeout(r, intervalMs));
        }

        if (!isMounted) return;

        // 🚨 타임아웃 되거나 백엔드가 죽었을 때 (HDBSCAN 에러 등)
        if (!clusters || clusters.length === 0) {
          console.warn("⚠️ 타임아웃 또는 분석 실패! 더미 데이터를 표시합니다.");
          setTotalArticles(3240);
          setClusterDetails(DUMMY_CLUSTERS);
          
          // 더미 타임라인 세팅 (타임라인 탭용)
          const dummyTimeline = [
            { id: 1, date: '02.06', volume: 20, phase: 'Phase 1', title: '의대 증원 발표', desc: '정부 공식 브리핑.', url: '#' },
            { id: 2, date: '02.20', volume: 100, phase: 'Phase 2', title: '전공의 사직', desc: '의료 대란 본격화.', url: '#' }
          ];
          setTimelineData(dummyTimeline);
          setActiveTimeline(dummyTimeline[1]);

          setDistributionData(DUMMY_MEDIA_DIST.map((item, idx) => ({
            ...item, left: ((item.center_x + 1)/2)*100, top: ((1 - item.center_y)/2)*100, size: 100, color: '#3b82f6'
          })));
          return; 
        }

        const total = clusters.reduce((sum, c) => sum + (Number(c.cluster_count) || 0), 0);
        setTotalArticles(total);
        setClusterDetails(clusters);

        // [추가된 타임라인 로직]
        try {
          const timelineJson = await fetchAPI(`/api/queries/${currentQueryId}/timeline`);
          const timelines = timelineJson?.data?.timelines || timelineJson?.timelines || [];
          if (timelines.length > 0) {
            const formattedTimeline = timelines.map((item, index) => {
              const d = new Date(item.timeline_date || Date.now());
              return {
                id: item.id || index,
                date: `${d.getMonth() + 1}.${String(d.getDate()).padStart(2, '0')}`,
                title: item.article_title, url: item.article_url, volume: 40 + Math.random() * 40,
                desc: item.summary || "", phase: index === 0 ? "Initial" : ""
              };
            });
            setTimelineData(formattedTimeline);
            setActiveTimeline(formattedTimeline[0]);
          } else {
             const dummyTimeline = [
              { id: 1, date: '02.06', volume: 20, phase: 'Phase 1', title: '의대 증원 발표', desc: '정부 공식 브리핑.', url: '#' },
              { id: 2, date: '02.20', volume: 100, phase: 'Phase 2', title: '전공의 사직', desc: '의료 대란 본격화.', url: '#' }
            ];
            setTimelineData(dummyTimeline);
            setActiveTimeline(dummyTimeline[1]);
          }
        } catch (e) { console.warn("타임라인 로드 실패", e); }


        try {
          const newFeatureMaps = {};
          for (const c of clusters) {
            const fmRes = await fetchAPI(`/api/queries/${currentQueryId}/clusters/${c.cluster_label}/feature-map`);
            const fm = fmRes?.data?.feature_map || fmRes?.feature_map;
            
            if (fm) {
              const nodes = [];
              const threshold = 0.3; 

              nodes.push({ id: `p_frame_${c.cluster_label}`, label: '보도 프레임', type: 'primary', top: '20%', left: '30%', color: '#3b82f6', desc: '기사의 주요 프레임' });
              nodes.push({ id: `p_logic_${c.cluster_label}`, label: '논조', type: 'primary', top: '70%', left: '60%', color: '#10b981', desc: '기사의 논리적 스탠스' });
              nodes.push({ id: `p_emo_${c.cluster_label}`, label: '감정선', type: 'primary', top: '40%', left: '80%', color: '#ec4899', desc: '내포된 주요 감정' });

              const addSubNodes = (items, parentId, startTop, startLeft, color) => {
                if (!items) return;
                items.filter(item => item.ratio > threshold).forEach((item, idx) => {
                  const words = item.loaded_words ? item.loaded_words.join(', ') : (item.stance_score ? `${item.stance_score.label} (${item.stance_score.value})` : '');
                  nodes.push({
                    id: `${parentId}_sub_${idx}`,
                    label: item.label,
                    type: 'secondary',
                    ratio: item.ratio, 
                    top: `${startTop + (idx * 15)}%`,
                    left: `${startLeft + (idx * 12)}%`,
                    color: color,
                    desc: `[비중: ${Math.round(item.ratio * 100)}%] \n${words ? `주요 단어: ${words}` : ''}`
                  });
                });
              };

              addSubNodes(fm.frame, `p_frame_${c.cluster_label}`, 5, 10, '#3b82f6');
              addSubNodes(fm.logic, `p_logic_${c.cluster_label}`, 55, 45, '#10b981');
              addSubNodes(fm.emotions, `p_emo_${c.cluster_label}`, 25, 75, '#ec4899');

              newFeatureMaps[c.cluster_label] = nodes;
            }
          }
          if (Object.keys(newFeatureMaps).length > 0) setFeatureMapsData(newFeatureMaps);
        } catch (e) { console.warn("피처맵 로드 실패", e); }

        try {
          const biasJson = await fetchAPI(`/api/queries/${currentQueryId}/bias-plane`);
          const mediaDist = biasJson?.data?.media_distribution || biasJson?.media_distribution || [];
          const colors = ['#ec4899', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6'];
          
          if (mediaDist.length > 0) {
            const mappedDist = mediaDist.map((item, idx) => {
              const leftPos = ((item.center_x + 1) / 2) * 100;
              const topPos = ((1 - item.center_y) / 2) * 100;
              const bubbleSize = 80 + (item.spread_radius * 400) + ((item.article_count || 1) * 3);

              return {
                ...item,
                left: leftPos,
                top: topPos,
                size: bubbleSize,
                color: colors[idx % colors.length]
              };
            });
            setDistributionData(mappedDist);
            setMatchedArticle(mappedDist[0] ? { press: mappedDist[0].source_name, title: "검색 기준과 가장 유사한 언론사" } : null);
          }
        } catch (e) { console.warn("좌표평면 로드 실패", e); }

      } catch (error) {
        console.error("데이터 세팅 중 오류:", error);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchAllData();
    return () => { isMounted = false; };
  }, [searchTerm]);

  const handleBiasSearch = async () => {
    // 1. 슬라이더의 현재 X, Y 값을 가져옵니다. (명세서 보니 60, 31 같은 정수 형태네요)
    const queryX = Math.round(biasX);
    const queryY = Math.round(biasY);

    try {
      // 2. 백엔드의 매칭 API를 호출합니다! (queryId는 이미 상태로 저장되어 있음)
      const matchData = await fetchAPI(`/api/queries/${queryId}/bias-plane/match?x=${queryX}&y=${queryY}`);
      
      // 3. 응답이 성공적으로 오면 화면을 업데이트합니다.
      if (matchData && matchData.data && matchData.data.matched_article) {
        const article = matchData.data.matched_article;
        
        setMatchedArticle({
          // 주의: API 응답에 언론사 이름(source_name)이 없어서 임시로 텍스트를 넣었습니다. 
          // 백엔드에 source_name도 같이 달라고 요청하시면 완벽합니다!
          press: "매칭된 기사", 
          title: article.title, // API에서 준 기사 제목
          url: article.url      // API에서 준 기사 링크
        });
      }
    } catch (error) {
      console.error("매칭 기사 검색 실패:", error);
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
          <p className="result-meta">• {isLoading ? '서버 분석 요청 중...' : `${(totalArticles||0).toLocaleString()}개의 기사 기반 분석 완료`}</p>
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
              <p style={{ marginTop: '20px', color: '#666' }}>데이터를 수집하고 분석 중입니다... (10초 주기)</p>
            </div>
          ) : (
            <>
              {/* ◼️ 탭 1: 기사 요약 */}
              {activeTab === 'summary' && (
                <div className="tab-summary fade-in">
                  <div className="summary-cards-container">
                    {clusterDetails.map((cluster, index) => (
                      <div className="editorial-summary-card" key={index}>
                        <div className="esc-left">
                          <div className="esc-header">
                            <span className={`esc-badge ${index === 0 ? 'primary' : 'secondary'}`}>{index === 0 ? 'Mainstream' : 'Minority'}</span>
                            <h3 className="esc-title">"{cluster.cluster_title}"</h3>
                          </div>
                          <div className="esc-body">
                            <p className="esc-paragraph">
                              <span className="drop-cap">{cluster.cluster_summary?.charAt(0) || '분'}</span>
                              {cluster.cluster_summary?.slice(1) || '석된 요약이 없습니다.'}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ◼️ 탭 2: 타임라인 */}
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
                          <div className="detail-header"><span className="p-phase-badge">{activeTimeline.phase}</span><span className="p-date-badge">{activeTimeline.date}</span></div>
                          <div className="timeline-text-content" style={{marginTop:'20px'}}>
                            <h3 className="d-article-title">{activeTimeline.title}</h3>
                            <p className="d-press-desc">{activeTimeline.desc}</p>
                          </div>
                          <button className="editorial-read-btn glass-btn" onClick={() => window.open('#')}>원문 읽기 →</button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ◼️ 탭 3: 피처맵 */}
              {activeTab === 'feature' && (
                <div className="tab-feature fade-in">
                  <div className="map-sub-nav" style={{display:'flex', gap:'10px', marginBottom:'20px'}}>
                    {clusterDetails.map((c, i) => (
                      <button key={i} className={`sub-tab-btn ${selectedCluster === c.cluster_label ? 'active' : ''}`} onClick={() => setSelectedCluster(c.cluster_label)}>관점 {i+1} 분석</button>
                    ))}
                  </div>

                  <div className="strict-grid-layout">
                    <div className="glass-card feature-map-bg" style={{minHeight:'600px', position:'relative', overflow:'hidden'}}>
                      {currentNodes.length > 0 ? currentNodes.map((node) => {
                        const dynamicSize = node.type === 'primary' ? '130px' : `${(node.ratio * 200) + 50}px`;
                        
                        return (
                          <div 
                            key={node.id} 
                            className={`f-node ${node.type} ${activeFeature?.id === node.id ? 'active' : ''}`} 
                            style={{ 
                              top: node.top, 
                              left: node.left, 
                              width: dynamicSize,
                              height: dynamicSize,
                              backgroundColor: node.type === 'primary' ? '#111' : 'rgba(255,255,255,0.95)',
                              border: node.type === 'primary' ? `3px solid ${node.color}` : `2px solid ${node.color}`,
                              color: node.type === 'primary' ? '#fff' : '#111',
                              boxShadow: activeFeature?.id === node.id ? `0 0 20px ${node.color}55` : 'none'
                            }} 
                            onClick={() => setActiveFeature(node)}
                          >
                            <span style={{fontWeight:'700', textAlign:'center', fontSize: node.type === 'primary' ? '1rem' : '0.85rem'}}>{node.label}</span>
                          </div>
                        );
                      }) : <p style={{padding:'20px'}}>선택된 군집의 피처맵 데이터가 없습니다.</p>}
                    </div>

                    <div className="glass-card insight-panel">
                      <span className="insight-badge">FEATURE DETAIL</span>
                      {activeFeature ? (
                        <>
                          <h2 className="press-name" style={{marginTop:'20px', color: activeFeature.color}}>{activeFeature.type === 'primary' ? '분석 카테고리' : '상세 키워드'}</h2>
                          <h3 className="article-title" style={{fontSize:'1.4rem'}}>{activeFeature.label}</h3>
                          <p className="press-desc" style={{marginTop:'15px', whiteSpace: 'pre-line', lineHeight: '1.6'}}>{activeFeature.desc}</p>
                        </>
                      ) : (
                        <p style={{marginTop:'40px', color:'#999'}}>버블을 클릭하시면 상세 분석 결과가 표시됩니다.</p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* ◼️ 탭 4: 좌표 평면 */}
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
                        
                        {distributionData.map((media, idx) => {
                          const topPos = mapSubTab === '2d' ? media.top : 50;
                          const textOffset = mapSubTab === '1d' ? (idx % 2 === 0 ? -45 : 45) : 0;
                          return (
                            <React.Fragment key={idx}>
                              <div className={`dist-blob ${mapSubTab === '1d' ? 'bubble-mode' : ''}`} 
                                style={{ 
                                  left: `${media.left}%`, top: `${topPos}%`, 
                                  width: `${media.size}px`, height: `${media.size}px`, 
                                  background: media.color, opacity: 0.6 
                                }}
                              ></div>
                              <span className="dist-text" style={{ left: `${media.left}%`, top: `calc(${topPos}% + ${textOffset}px)` }}>{media.source_name}</span>
                            </React.Fragment>
                          );
                        })}
                      </div>
                    </div>
                    <div className="right-sidebar">
                      <div className="glass-card slider-control-panel">
                        <h3 className="control-title">TARGET SETTING</h3>
                        <input type="range" min="0" max="100" value={biasX} onChange={(e)=>setBiasX(e.target.value)} className="aesthetic-slider" />
                        <button className="search-target-btn" onClick={handleBiasSearch}>기사 찾기</button>
                      </div>
                      <div className="glass-card insight-panel map-insight">
                        {matchedArticle ? (
                          <>
                            <span className="insight-badge">검색 완료</span>
                            
                            {/* 실제 기사 제목 출력 */}
                            <h3 className="article-title" style={{ wordBreak: 'keep-all', lineHeight: '1.4' }}>
                              {matchedArticle.title}
                            </h3>
                            
                            {/* 기사 원문 읽기 버튼 */}
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