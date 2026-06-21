import ReactMarkdown, { type Components } from "react-markdown";
import "./Markdown.css";

const components: Components = {
  h1: ({ children }) => <p className="md-heading">{children}</p>,
  h2: ({ children }) => <p className="md-heading">{children}</p>,
  h3: ({ children }) => <p className="md-heading">{children}</p>,
  h4: ({ children }) => <p className="md-heading">{children}</p>,
  hr: () => <hr className="md-rule" />,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
  code: ({ children }) => <code className="mono">{children}</code>,
};

export default function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown components={components}>{content}</ReactMarkdown>
    </div>
  );
}
