import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ChartComponent from './ChartComponent';

export default function MessageContent({ content }: { content: string }) {
    if (content.startsWith("{ \"type\": \"chart\"")) {
        const payload = JSON.parse(content);
        return <ChartComponent data={payload.data} />;
    }

    return (
        <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            components={{
                // 1. Wrap the table in a responsive, scrolling container block
                table: ({ node, ...props }) => (
                    <div className="md-table-wrap">
                        <table className="md-table" {...props} />
                    </div>
                ),
                thead: ({ node, ...props }) => (
                    <thead className="md-thead" {...props} />
                ),
                th: ({ node, ...props }) => (
                    <th className="md-th" {...props} />
                ),
                td: ({ node, ...props }) => (
                    <td className="md-td" {...props} />
                ),
                tr: ({ node, ...props }) => (
                    <tr className="md-tr" {...props} />
                ),
            }}
        >
            {content}
        </ReactMarkdown>
    );
}